import argparse
from pathlib import Path
from sgld_cv.data import download, prepare_year

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download and checksum UCI data; optionally prepare Year training rows")
    p.add_argument("dataset", choices=["german", "year"])
    p.add_argument("--cache", type=Path, default=Path("data/raw"))
    p.add_argument("--prepare", type=Path)
    p.add_argument("--rows", type=int, default=463715)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--intercept", action="store_true")
    a = p.parse_args()
    archive = download(a.dataset, a.cache)
    print(archive, flush=True)
    if a.prepare:
        if a.dataset != "year": p.error("--prepare applies to Year only")
        print(prepare_year(archive, a.prepare, rows=a.rows, standardize=a.standardize, intercept=a.intercept))
