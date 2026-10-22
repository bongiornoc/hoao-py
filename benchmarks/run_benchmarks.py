"""Reproducible CPU timing and process memory for complete public operations.

Run ``python benchmarks/run_benchmarks.py --pairs 100 --output results.json``.
Each case runs in a fresh subprocess with one BLAS thread. Peak RSS includes
the interpreter, imported libraries, synthetic data and all timed operations.
Results are hardware observations, not performance guarantees.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def worker(n: int, train_size: int, pairs: int) -> dict:
    import numpy as np
    import scipy
    from hoao import CV, ReturnsPanel, calibrate

    rng = np.random.default_rng(43)
    universe = max(250, 2 * n)
    rows = 1800
    history = rng.standard_normal((rows, universe)) * .01
    history += rng.standard_normal((rows, 1)) * .006
    current = history[-train_size:, :n]
    started = time.perf_counter()
    panel = ReturnsPanel(history)
    panel.eligible(train_size=train_size)
    panel_seconds = time.perf_counter() - started
    # Warm up LAPACK and BLAS before measuring estimation throughput.
    CV(random_state=4).fit(current)
    started = time.perf_counter()
    cv_model = CV(random_state=4).fit(current)
    cv_seconds = time.perf_counter() - started
    started = time.perf_counter()
    calibration = calibrate(panel, n_assets=n, train_size=train_size,
                            test_size=100, n_pairs=pairs, before=1600,
                            random_state=5, n_jobs=1)
    calibration_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = calibration.fit(current, as_of=1799)
    apply_seconds = time.perf_counter() - started
    started = time.perf_counter()
    allocation = model.optimize(gamma=10)
    optimization_seconds = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    return dict(n_assets=n, train_size=train_size, test_size=100,
                universe_size=universe, panel_rows=rows, n_pairs=pairs,
                panel_seconds=panel_seconds, cv_seconds=cv_seconds,
                calibration_seconds=calibration_seconds,
                calibration_pairs_per_second=pairs/calibration_seconds,
                apply_seconds=apply_seconds, optimization_seconds=optimization_seconds,
                gradient_residual=allocation.gradient_residual,
                peak_process_rss_mib=peak_mib,
                compact_fourth_bytes=sum(a.nbytes for a in [cv_model.cokurtosis.basis,
                                                           cv_model.cokurtosis.core,
                                                           cv_model.cokurtosis.scales]),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", nargs=2, type=int, metavar=("ASSETS", "TRAIN"))
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.case:
        print(json.dumps(worker(*args.case, args.pairs)))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    results = []
    for n in (50, 100, 200):
        for train in (100, 500):
            command = [sys.executable, __file__, "--case", str(n), str(train), "--pairs", str(args.pairs)]
            row = json.loads(subprocess.check_output(command, env=env, text=True))
            results.append(row)
            print(f"n={n}, train={train}: CV {row['cv_seconds']:.3f}s, "
                  f"calibration {row['calibration_pairs_per_second']:.1f} pairs/s, "
                  f"peak {row['peak_process_rss_mib']:.1f} MiB", flush=True)
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    report = dict(cpu=cpu, platform=platform.platform(), blas_threads=1,
                  gpu="Not measured: CUDA driver unavailable on benchmark host.", cases=results)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
