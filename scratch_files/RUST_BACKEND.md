# Rust backend feasibility and measured prototype

This is an investigation with runnable native kernels and reproducible benchmarks. It is **not a complete Rust replacement** for Bayesian Filters. Existing imports and filter classes still use the unchanged NumPy/SciPy implementation. The optional `bayesian_filters.rust` module exposes the experimental subset explicitly.

## Build and reproduce

Requires Python 3.11+, a recent stable Rust toolchain, and a C linker. The extension is a separate distribution so ordinary users do not suddenly need Rust to install Bayesian Filters.

```bash
uv venv
uv pip install -e '.[dev]' 'maturin>=1.9,<2'
uv run --no-sync maturin build --release --locked --manifest-path rust/Cargo.toml --interpreter .venv/bin/python --out native-wheels
uv pip install native-wheels/*.whl
uv run --no-sync pytest scratch_files/rust_backend/test_native.py --benchmark-disable
uv run --no-sync python scratch_files/rust_backend/benchmark.py
BENCHMARK_THREADS=4 uv run --no-sync python scratch_files/rust_backend/benchmark.py --output scratch_files/rust_backend/results_parallel.json
uv run --no-sync python scratch_files/rust_backend/report.py
```

Do not benchmark a debug build. Python-call timings include dtype conversion, validation, allocation, and the FFI boundary. Large matrix results compare against the actual installed NumPy BLAS backend, not interpreted matrix loops. The benchmark pins one available CPU and limits BLAS/OpenMP to one thread. Compilation and installation are outside the timing. Nine paired rounds alternate execution order, following warmup and calibration; raw samples, environment, and numerical errors are saved. This is a Linux, CPU-only experiment on one development host, not a cross-platform performance guarantee.

[Measured results](rust_backend/results.json), [standalone HTML report](rust_backend/report.html), and [complete source inventory](rust_backend/inventory.json) are generated from the checked-in scripts. The HTML includes vectorized NumPy controls where applicable: those changes may be preferable to adding a native dependency.

## Explicit backend selection

```python
from bayesian_filters.backends import get_backend

backend = get_backend("numpy")  # default; native extension is not required
backend = get_backend("rust")   # explicit opt-in; never changes other imports
posterior = backend.discrete_update(likelihood, prior)
```

Both expose the same limited kernel entry points. The original implementation files
are untouched. Selecting Rust does not silently change existing KalmanFilter objects
or claim native coverage for algorithms outside this prototype.

## SIMD and parallel execution

Large matrix products use `gemm` with runtime SIMD dispatch (including AVX-512 when
available), while short probability/vector operations use `wide` SIMD lanes. Circular
convolution splits wrapped regions into contiguous SIMD operations. `Rayon` parallelizes
large GEMMs and large probability/convolution chunks; sequential g-h and Kalman time
steps retain their dependency ordering.

Rust defaults to one worker. Set `BAYESIAN_FILTERS_RUST_THREADS=4` **before the first
native call** to use the bounded persistent pool (1–32 threads). Work thresholds avoid
launching parallel work for tiny inputs. The extension currently holds the Python GIL;
Rayon workers run native operations, not Python callbacks. No `target-cpu=native` flag is
needed, so the build does not require the consumer to share the build host's SIMD ISA.

Run matched four-thread comparisons with:

```bash
BENCHMARK_THREADS=4 uv run --no-sync python scratch_files/rust_backend/benchmark.py --output scratch_files/rust_backend/results_parallel.json
uv run --no-sync python scratch_files/rust_backend/report.py
```

The report retains the original prototype measurements in `results_initial.json` and
compares them with the optimized implementation. Thread limits apply equally to BLAS
and Rust in each run. Existing NumPy implementations, including vectorized controls,
are retained even when Rust wins; this work adds a choice rather than replacing code.

## What executes in Rust

- Linear Kalman predict/update using dense `f64` nalgebra matrices and Joseph covariance updates. The fused batch API returns both prior and posterior states/covariances. It requires fixed F/Q/H/R, flat states and finite measurements.
- Default unscented weighted moments. Existing UKF instances can opt in through `predict(UT=rust.unscented_transform)` and `update(..., UT=rust.unscented_transform)`. Custom means/residuals intentionally use the Python reference; Python fx/hx callbacks and sigma-point generation remain Python.
- Scalar g-h batch filtering, including predictions and the initial state, matching `GHFilter.batch_filter` without Saver hooks.
- One-dimensional discrete update and integer-offset, wrap-mode prediction.
- Systematic and stratified resampling traversal. Random numbers still come from NumPy, preserving its seeded sequence. Index outputs use int64 rather than the reference's int32.

Shape checks prevent invalid dimensions from reaching matrix operations. Probability kernels reject invalid inputs rather than silently normalizing negative or zero-mass distributions. These stricter error contracts, and `ValueError` for singular Kalman innovation matrices, are another reason the prototype is opt-in rather than a silent replacement.

## What an all-numerical-Rust backend entails

The inventory lists every implementation module and public class/function. Porting the full numerical backend is feasible, but a transparent replacement must cover these separate compatibility contracts:

| Area | Full-backend requirements |
| --- | --- |
| Linear Kalman and fading variants | Mutable x/P/F/Q/H/R/B views, scalar and column-vector shapes, controls, alpha, missing measurements, variable matrices, correlated/sequential/steady-state updates, custom inverse, diagnostics and Saver hooks. The prototype supports only ordinary fixed-matrix steps/batches. |
| EKF, UKF and cubature | Preserve Python fx/hx/Jacobian/custom mean/residual callbacks, argument forwarding, ordering and exception propagation. Native kernels alone do not eliminate callback overhead. |
| Sigma-point families | Merwe, Julier and Simplex weights/ordering, custom subtraction and square-root functions, negative weights and conditioning. |
| Square-root and information filters | Stable QR/Cholesky and information-form algebra, singular cases and established diagnostics. These are not interchangeable with the ordinary Kalman kernel. |
| RTS and fixed-lag smoothers | Full trajectory outputs, gains, prior covariances, variable models and lag boundaries. |
| EnKF, IMM and MMAE | RNG consumption, Python subfilter objects, mixing probabilities, likelihood normalization and object-state synchronization. |
| GH/GHK, least squares and fading memory | Scalar/vector variants, update order, error/variance estimates and complete batch/state contracts. |
| Discrete Bayes and particle resampling | In-place normalization, non-wrap boundaries, fractional shift behavior, residual/multinomial semantics and seeded RNG compatibility. |
| Statistics and common helpers | Probability distributions, singular covariance behavior, matrix exponential discretization, noise matrices, kinematic builders and public shapes. Plotting, representation and inspection utilities should stay Python. |

The design I recommend is a stable Python facade over native stateful filters and fused sequence APIs, with explicit callback boundaries and optimized dense linear algebra. Preserve NumPy array interchange, keep the reference backend selectable, and require parity on every advertised API before changing defaults. Packaging needs tested wheels for supported Python versions, Linux/macOS/Windows and common architectures; the local experimental wheel is not that distribution matrix.

Pure Rust source does not itself guarantee faster linear algebra. The prototype deliberately reports slowdowns as well as wins. Handwritten covariance loops should not replace optimized BLAS on large states, and simple NumPy vectorization can remove some current Python overhead without a Rust port. Decide kernel dispatch and matrix-library choices from these measurements instead of applying a blanket replacement.

## Validation evidence

The unchanged reference suite passes 231 tests. The optional native suite passes 50
additional differential/edge-case tests with both one and four Rust workers. Release
wheel building and `cargo clippy --release -- -D warnings` pass. The interactive HTML
report was checked in a browser, including thread selection and workload filtering.

See the [measured comparison table](rust_backend/summary.md) for the 31 workloads
measured at each thread count. Performance does not justify silently replacing all
existing implementations: the dense Kalman cases remain a measured limitation.

## Correctness boundaries

Differential tests compare fixed-seed trajectories, priors/posteriors, negative sigma weights, callback fallbacks, even/odd convolution kernels, negative/large offsets, noncontiguous/read-only inputs, seeded resampling indices, empty batches and malformed inputs. Benchmark parity is checked before timings at `rtol=atol=2e-9`; raw maximum errors are recorded.

The existing 231-test suite remains the baseline for APIs outside the prototype. It passing does not establish complete native API coverage: those tests still exercise the reference backend. The dedicated native tests establish the limited native contracts above. No external upstream repository is modified or targeted by this work.
