# Second-order control variates for SGLD

Research code and manuscript for **Second-Order Control Variates for Stochastic Gradient Langevin Dynamics**.

**Authors:** Michelle Le, Frank Fan, Jay Chooi, and Alexander Young.

[Read the manuscript](paper/main.pdf) · [LaTeX source](paper/main.tex)

The experiments study when second-order control variates improve posterior accuracy and measured time-to-accuracy relative to first-order control variates and full-gradient Langevin sampling. Main comparisons use a fixed full-Hessian preconditioner. Benefits depend on the target, observation-level Taylor remainders, accuracy requirements, tuning, setup costs, and implementation; the experiments do not establish universal superiority.

## Repository layout

- `paper/`: the supplied manuscript and its matching LaTeX source, bibliography, tables, and vector figures.
- `src/sgld_cv/`: models, gradient estimators, samplers, preconditioners, transport metrics, diagnostics, and cost accounting.
- `configs/`, `scripts/`, `experiments/`: experiment configurations and entry points.
- `gpu_replication/`, `p2_screening/`, `p2_synthetic/`, `p3a/`, `p3b/`, `p4/`, `p5/`: backend and staged experiment source code.
- `tests/`: numerical and protocol checks.

## Install and run numerical checks

Python 3.10 or later:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest tests/test_core.py tests/test_gaussian.py
```

`requirements-validated.txt` records the versions used for the original local validation. JAX implementations additionally require an appropriate JAX installation; GPU support depends on the runtime and hardware.

## Compile the paper

From `paper/`, with a LaTeX installation providing pdfLaTeX and latexmk:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source produces one document containing the paper and Appendices A.1–A.11. It can also be uploaded to Overleaf with `main.tex` as the main document.

## Data and experiment archives

This repository contains the source code and manuscript compilation assets. Large datasets, saved reference pools, generated runs, checkpoints, local environments, and machine-specific backup files are not included. Some experiment scripts and archive-dependent tests require those assets and the original project-relative paths. This initial source release is not a complete reproduction archive for every reported run.

The provided PDF is a research manuscript, not a claim of publication or proof certification.
