# Measured backend comparisons

Release build on AMD EPYC 7B13; Python 3.13.12.
Nine paired rounds; times include the Python boundary. Existing NumPy/SciPy code remains the default.
Rust is faster above 1x. Thread counts are matched between BLAS and Rust within each run.

| Workload | 1-thread speedup | 4-thread speedup |
| --- | ---: | ---: |
| kalman step n=2 m=1 | 8.59x | 8.53x |
| kalman step n=4 m=2 | 7.63x | 8.17x |
| kalman step n=16 m=4 | 3.54x | 3.56x |
| kalman step n=64 m=16 | 0.92x | 0.91x |
| kalman step n=128 m=32 | 0.79x | 0.49x |
| kalman step n=256 m=64 | 0.68x | 0.40x |
| kalman batch n=2 m=1 T=1000 | 80.35x | 83.06x |
| kalman batch n=4 m=2 T=1000 | 42.92x | 46.95x |
| kalman batch n=16 m=4 T=1000 | 7.60x | 6.81x |
| kalman batch n=32 m=8 T=1000 | 3.03x | 3.22x |
| unscented transform n=2 | 3.49x | 3.51x |
| unscented transform n=8 | 2.66x | 2.48x |
| unscented transform n=32 | 1.67x | 1.65x |
| unscented transform n=128 | 1.69x | 0.91x |
| unscented transform n=256 | 1.82x | 1.36x |
| UKF full sequence n=4 T=200 | 1.09x | 1.16x |
| UKF full sequence n=16 T=200 | 1.04x | 1.03x |
| GH batch T=100 | 41.81x | 42.66x |
| GH batch T=10000 | 95.09x | 97.65x |
| systematic resample N=100 | 5.93x | 5.89x |
| discrete update N=100 | 8.52x | 8.79x |
| discrete predict N=100 | 17.46x | 17.66x |
| systematic resample N=10000 | 31.81x | 31.40x |
| discrete update N=10000 | 68.55x | 69.55x |
| discrete predict N=10000 | 5.74x | 5.76x |
| systematic resample N=100000 | 25.27x | 25.11x |
| discrete update N=100000 | 74.23x | 71.62x |
| discrete predict N=100000 | 4.11x | 4.50x |
| systematic resample N=1000000 | 22.14x | 25.49x |
| discrete update N=1000000 | 36.73x | 159.39x |
| discrete predict N=1000000 | 1.40x | 6.46x |

Large dense Kalman steps still favor the existing BLAS path on this host despite SIMD/GEMM and Rayon. Vectorized NumPy controls can also beat native kernels; see the interactive report for that comparison. The optional backend is a measured subset, not a full native port.

Maximum absolute discrepancy across these cases: 9.21e-15; all cases pass rtol=atol=2e-9 before timing. This does not establish parity for unported APIs.

[Interactive report](report.html) · [Raw single-thread results](results.json) · [Raw parallel results](results_parallel.json) · [API inventory](inventory.json)
