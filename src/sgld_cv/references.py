"""Analytic ridge posteriors and Metropolis-corrected, whitened full-data HMC.

References are evaluation infrastructure. Ridge sufficient-statistic costs are
also attributed to methods which use the resulting mode/precision.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from .anchors import supplied_anchor
from .costs import CostLedger
from .diagnostics import chain_diagnostics
from .randomness import generator


@dataclass
class RidgePosterior:
    mean: np.ndarray
    precision: np.ndarray
    chol_precision: np.ndarray
    cost: CostLedger

    @property
    def covariance(self):
        return cho_solve((self.chol_precision, True), np.eye(len(self.mean)))

    def sample(self, size, rng):
        return self.mean + solve_triangular(self.chol_precision.T,
                                           rng.normal(size=(len(self.mean), size)), lower=False).T


def ridge_posterior(model, chunk_size=8192):
    cost = CostLedger()
    with cost.measure("anchor_sufficient_statistics"):
        Q = model.prior_precision * np.eye(model.d)
        rhs = np.zeros(model.d)
        for first in range(0, model.N, chunk_size):
            X = model.X[first:first + chunk_size]
            y = model.y[first:first + chunk_size]
            Q += X.T @ X / model.noise_variance
            rhs += X.T @ y / model.noise_variance
            cost.count("sufficient_statistic_rows", len(X))
            cost.count("data_row_visits", len(X))
        L = np.linalg.cholesky(Q)
        mean = cho_solve((L, True), rhs)
        cost.count("cholesky_factorizations")
        cost.count("linear_solves")
    return RidgePosterior(mean, Q, L, cost)


class CachedGaussianEstimator:
    """Full-gradient ULA on Gaussian sufficient statistics; no data per step."""
    method = "ula"
    def __init__(self, model, posterior):
        self.model, self.posterior = model, posterior
        self.setup_cost = posterior.cost.copy()
        self.cache_bytes = posterior.precision.nbytes + posterior.mean.nbytes

    def gradient(self, theta, indices=None, cost=None, *, return_raw_mean=False):
        g = self.posterior.precision @ (theta - self.posterior.mean)
        if cost is not None:
            cost.count("dense_precision_vector_products")
        return (g, (g - self.model.prior_gradient(theta)) / self.model.N) if return_raw_mean else g


def hmc_reference(model, anchor, *, chains=4, warmup=500, draws=1000, seed=7000,
                  step_size=.3, leapfrog_steps=12):
    """Random-length HMC; step adaptation only during discarded warmup.

    theta = anchor + L^-T z, L L.T = Hessian(anchor). Unit mass in z.
    A Laplace Hessian changes coordinates only; MH targets the actual posterior.
    """
    if chains < 2 or warmup < 1 or draws < 4 or step_size <= 0 or leapfrog_steps < 2:
        raise ValueError("invalid reference configuration")
    start = perf_counter()
    L = np.linalg.cholesky(model.full_hessian(anchor.theta))
    transform = solve_triangular(L.T, np.eye(model.d), lower=False)
    samples = np.empty((chains, draws, model.d))
    acceptance, divergences, epsilons = [], [], []
    gradient_calls = 0
    def evaluate(z):
        nonlocal gradient_calls
        theta = anchor.theta + transform @ z
        gradient_calls += 1
        return model.potential(theta), transform.T @ model.full_gradient(theta)
    for chain in range(chains):
        rng = generator(seed, chain, 70)
        z = rng.normal(size=model.d) * 1.5
        eps = step_size
        accepted = div = 0
        U, gradient = evaluate(z)
        for iteration in range(warmup + draws):
            momentum = rng.normal(size=model.d)
            p, q = momentum.copy(), z.copy()
            epsilon = eps * rng.uniform(.9, 1.1)
            length = int(rng.integers(max(2, leapfrog_steps // 2), leapfrog_steps + 1))
            with np.errstate(over="ignore", invalid="ignore"):
                p -= epsilon * gradient / 2
                for j in range(length):
                    q += epsilon * p
                    U_new, gradient_new = evaluate(q)
                    p -= epsilon * gradient_new * (0.5 if j == length - 1 else 1)
                energy_error = U_new + .5 * (p @ p) - U - .5 * (momentum @ momentum)
            probability = float(np.exp(min(0, -energy_error))) if np.isfinite(energy_error) else 0.0
            accept = rng.random() < probability
            if accept:
                z, U, gradient = q, U_new, gradient_new
            if iteration < warmup:
                logeps = np.log(eps) + .05 / np.sqrt(1 + iteration / 10) * (probability - .8)
                eps = float(np.clip(np.exp(logeps), .001, 1.5))
            else:
                accepted += int(accept)
                div += int(not np.isfinite(energy_error) or abs(energy_error) > 100)
                samples[chain, iteration - warmup] = anchor.theta + transform @ z
        acceptance.append(accepted / draws)
        divergences.append(div)
        epsilons.append(eps)
    diagnostic = chain_diagnostics(samples)
    diagnostic.update({"kind": "full-data Metropolis HMC, approximate reference",
        "acceptance": acceptance, "divergences": divergences, "final_step_sizes": epsilons,
        "seed": seed, "warmup": warmup, "reference_seconds": perf_counter() - start,
        "full_gradient_calls": gradient_calls})
    diagnostic["reliable"] &= not any(divergences) and min(acceptance) > .5
    return samples, diagnostic


def validate_reference_pair(first, second, first_diagnostic, second_diagnostic):
    """Independent longer run agreement supplements Rhat/ESS diagnostics."""
    mean1, mean2 = first.mean(axis=(0, 1)), second.mean(axis=(0, 1))
    se1 = np.array([r["mcse_mean"] or np.inf for r in first_diagnostic["coordinates"]])
    se2 = np.array([r["mcse_mean"] or np.inf for r in second_diagnostic["coordinates"]])
    z = np.abs(mean1 - mean2) / np.sqrt(se1 ** 2 + se2 ** 2)
    covariance1 = np.cov(first.reshape(-1, first.shape[-1]), rowvar=False)
    covariance2 = np.cov(second.reshape(-1, second.shape[-1]), rowvar=False)
    covariance_error = float(np.linalg.norm(covariance1 - covariance2) / np.linalg.norm(covariance2))
    return {"reliable": bool(first_diagnostic["reliable"] and second_diagnostic["reliable"] and
                              np.max(z) < 4 and covariance_error < .2),
            "max_mean_mcse_z": float(np.max(z)), "covariance_relative_difference": covariance_error,
            "limitations": "MCMC reference draws remain correlated; diagnostics are not a proof of exactness"}
