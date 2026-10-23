Measured performance
====================

The CPU implementation was measured on an Intel Core i7-11850H at 2.50 GHz,
running Linux x86-64, with Python 3.11.11, NumPy 2.4.6, and SciPy 1.17.1.
Each case ran in a fresh process with one BLAS thread. These measurements
describe the synthetic cases below; they are not throughput guarantees.

.. list-table:: Complete public operations, CPU
   :header-rows: 1

   * - Assets
     - Training rows
     - CV fit (s)
     - Calibration pairs/s
     - Apply core (s)
     - Peak process MiB
   * - 50
     - 100
     - 0.0087
     - 914.5
     - 0.0015
     - 100.0
   * - 50
     - 500
     - 0.0165
     - 404.6
     - 0.0032
     - 98.9
   * - 100
     - 100
     - 0.0334
     - 278.9
     - 0.0044
     - 99.7
   * - 100
     - 500
     - 0.0362
     - 202.3
     - 0.0053
     - 99.5
   * - 200
     - 100
     - 0.1033
     - 97.6
     - 0.0121
     - 113.8
   * - 200
     - 500
     - 0.1063
     - 80.0
     - 0.0144
     - 112.2

The return panel has 1,800 rows and a universe of 250 assets for the 50- and
100-asset cases, or 400 assets for the 200-asset cases. Returns combine
independent Gaussian noise with a common factor; all observations are finite.
The benchmark performs one warm-up CV fit, then times a default ten-split
CV fit, a 100-pair historical calibration with 100 testing rows per pair,
application of the resulting core, and a ``gamma=10`` portfolio optimization.
Calibration uses one CPU worker. A calibration's reported duration includes
sampling and preprocessing; panel construction and eligibility preparation
are measured separately.

Peak resident memory includes the Python interpreter, imported dependencies,
the synthetic panel, and all measured operations in that process. It is not
the incremental memory of a single fit. The filtered fourth moment's basis,
core and marginal-scale arrays occupied 40,400 bytes at 50 assets,
160,800 bytes at 100 assets, and 641,600 bytes at 200 assets. These array
counts exclude the cached volatility-scaled basis used for portfolio
calculations; it occupies another square matrix. A dense float64 fourth
tensor at 200 assets would require 12.8 billion bytes for the tensor alone.

Reproduce the measurements on a Linux or macOS host from the repository:

.. code-block:: console

   python benchmarks/run_benchmarks.py --pairs 100 --output benchmark-results.json

The script and unrounded results are in ``benchmarks/run_benchmarks.py`` and
``benchmarks/cpu-results.json``. There is one measured fit per case, not a
statistical timing distribution, and no comparison against another library.
Performance depends on BLAS, hardware, dimensions, and the chosen data.

The benchmark host had no available CUDA driver. GPU support is experimental,
and no GPU timing or speedup claim is made. Profile your actual workload:
transfers and kernel launches can dominate small fits, while large batches
may benefit from device matrix operations.
