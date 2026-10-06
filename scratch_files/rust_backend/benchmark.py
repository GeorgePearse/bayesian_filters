"""Paired end-to-end Python-call timings, fixed inputs, parity before timing."""

import os

THREADS = int(os.environ.get("BENCHMARK_THREADS", "1"))
if not 1 <= THREADS <= 32:
    raise ValueError("BENCHMARK_THREADS must be 1-32")
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BAYESIAN_FILTERS_RUST_THREADS"]:
    os.environ[name] = str(THREADS)
import hashlib
import argparse
import contextlib
import gc
import io
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
import scipy
from bayesian_filters import rust
from bayesian_filters.kalman import KalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter, unscented_transform
from bayesian_filters.gh import GHFilter
from bayesian_filters.discrete_bayes import predict, update
from bayesian_filters.monte_carlo import systematic_resample


def linear(n, m, t):
    rng = np.random.default_rng(20261006 + n)
    f = np.eye(n) * 0.95
    if n > 1:
        f[0, 1] = 0.1
    h = rng.normal(size=(m, n))
    q = np.eye(n) * 0.01
    r = np.eye(m) * 0.2
    x = rng.normal(size=n)
    p = np.eye(n)
    zs = rng.normal(size=(t, m))

    def ref():
        k = KalmanFilter(n, m)
        k.x = x.copy()
        k.P = p.copy()
        k.F = f
        k.H = h
        k.Q = q
        k.R = r
        return k.batch_filter(zs)

    return ref, lambda: rust.kalman_batch(x, p, zs, f, q, h, r)


def step(n, m):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    p = np.eye(n)
    f = np.eye(n) * 0.95
    q = np.eye(n) * 0.01
    r = np.eye(m) * 0.2
    h = rng.normal(size=(m, n))
    z = rng.normal(size=m)
    # Retain the class across calls; reset identical state before each step.
    k = KalmanFilter(n, m)
    k.F = f
    k.H = h
    k.Q = q
    k.R = r

    def ref():
        k.x = x.copy()
        k.P = p.copy()
        k.predict()
        k.update(z)
        return k.x, k.P

    return ref, lambda: rust.kalman_step(x, p, z, f, q, h, r)


def ukf(n, t):
    zs = np.random.default_rng(n).normal(size=(t, 2))

    def run(ut):
        points = MerweScaledSigmaPoints(n, 0.3, 2.0, 0.0)
        k = UnscentedKalmanFilter(n, 2, 0.1, lambda x: x[:2], lambda x, dt: x * 0.99, points)
        k.Q = np.eye(n) * 0.01
        k.R = np.eye(2) * 0.2
        out = []
        for z in zs:
            k.predict(UT=ut)
            k.update(z, UT=ut)
            out.append(k.x.copy())
        return np.array(out), k.P

    return lambda: run(unscented_transform), lambda: run(rust.unscented_transform)


def leaves(value):
    if isinstance(value, tuple):
        for v in value:
            yield from leaves(v)
    else:
        yield np.asarray(value)


def measure(name, ref, native, control=None):
    expected = list(leaves(ref()))
    actual = list(leaves(native()))
    assert len(expected) == len(actual)
    maximum = 0.0
    for a, b in zip(expected, actual):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-9)
        if a.size:
            maximum = max(maximum, float(np.max(np.abs(a - b))))
    funcs = {"numpy": ref, "rust": native}
    if control:
        for a, b in zip(expected, leaves(control())):
            np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-9)
        funcs["numpy_vectorized"] = control
    loops = {}
    samples = {k: [] for k in funcs}
    for key, fn in funcs.items():
        fn()
        start = time.perf_counter()
        fn()
        dt = time.perf_counter() - start
        loops[key] = max(1, min(10000, int(0.04 / max(dt, 1e-9))))
    gc.disable()
    try:
        for repeat in range(9):
            keys = list(funcs)
            if repeat % 2:
                keys.reverse()
            for key in keys:
                start = time.perf_counter_ns()
                for _ in range(loops[key]):
                    funcs[key]()
                samples[key].append((time.perf_counter_ns() - start) / loops[key] / 1000)
    finally:
        gc.enable()
    medians = {k: statistics.median(v) for k, v in samples.items()}
    result = {
        "case": name,
        "microseconds": medians,
        "samples_us": samples,
        "loops_per_sample": loops,
        "speedup": medians["numpy"] / medians["rust"],
        "max_abs_error": maximum,
    }
    if control:
        result["speedup_vs_vectorized"] = medians["numpy_vectorized"] / medians["rust"]
    print(json.dumps({k: v for k, v in result.items() if k not in {"samples_us", "loops_per_sample"}}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="scratch_files/rust_backend/results.json")
    args = p.parse_args()
    tool_env = os.environ.copy()
    tool_env.pop("LD_LIBRARY_PATH", None)
    compiler = subprocess.check_output(["rustc", "--version"], text=True, env=tool_env).strip()
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    results = []
    for n, m in [(2, 1), (4, 2), (16, 4), (64, 16), (128, 32), (256, 64)]:
        results.append(measure(f"kalman step n={n} m={m}", *step(n, m)))
    for n, m in [(2, 1), (4, 2), (16, 4), (32, 8)]:
        results.append(measure(f"kalman batch n={n} m={m} T=1000", *linear(n, m, 1000)))
    for n in [2, 8, 32, 128, 256]:
        rng = np.random.default_rng(n)
        s = rng.normal(size=(2 * n + 1, n))
        w = np.ones(2 * n + 1) / (2 * n + 1)
        q = np.eye(n) * 0.1

        def control():
            x = w @ s
            y = s - x
            return x, y.T @ (w[:, None] * y) + q

        results.append(
            measure(
                f"unscented transform n={n}",
                lambda: unscented_transform(s, w, w, q),
                lambda: rust.unscented_transform(s, w, w, q),
                control,
            )
        )
    for n in [4, 16]:
        results.append(measure(f"UKF full sequence n={n} T=200", *ukf(n, 200)))
    for n in [100, 10000]:
        data = np.random.default_rng(n).normal(size=n)
        gh = GHFilter(0.0, 0.0, 0.1, 0.2, 0.02)
        results.append(
            measure(
                f"GH batch T={n}", lambda: gh.batch_filter(data), lambda: rust.gh_batch(data, 0.0, 0.0, 0.1, 0.2, 0.02)
            )
        )
    for n in [100, 10000, 100000, 1000000]:
        rng = np.random.default_rng(n)
        w = rng.random(n)
        w /= w.sum()
        likelihood = rng.random(n)

        def ref():
            np.random.seed(42)
            return systematic_resample(w)

        def native():
            np.random.seed(42)
            return rust.systematic_resample(w)

        def vectorized():
            np.random.seed(42)
            positions = (np.random.random() + np.arange(n)) / n
            return np.searchsorted(np.cumsum(w), positions, side="right")

        results.append(measure(f"systematic resample N={n}", ref, native, vectorized))

        def update_control():
            posterior = w * likelihood
            return posterior / posterior.sum()

        results.append(
            measure(
                f"discrete update N={n}",
                lambda: update(likelihood, w),
                lambda: rust.discrete_update(likelihood, w),
                update_control,
            )
        )
        kernel = np.array([0.1, 0.7, 0.2])
        results.append(
            measure(
                f"discrete predict N={n}", lambda: predict(w, 2, kernel), lambda: rust.discrete_predict(w, 2, kernel)
            )
        )
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    report = {
        "method": {
            "seed": 20261006,
            "repetitions": 9,
            "target_seconds_per_sample": 0.04,
            "gc": "disabled during timing",
            "threads": THREADS,
            "cpu_affinity": list(os.sched_getaffinity(0)),
            "units": "microseconds per complete Python call, input conversion and output allocation included",
            "parity_rtol": 2e-9,
            "parity_atol": 2e-9,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "cpu": next(
                (
                    l.split(":", 1)[1].strip()
                    for l in Path("/proc/cpuinfo").read_text().splitlines()
                    if l.startswith("model name")
                ),
                "",
            ),
            "rustc": compiler,
            "native_source_sha256": hashlib.sha256(Path("rust/src/lib.rs").read_bytes()).hexdigest(),
            "profile": "release, thin LTO, portable runtime SIMD dispatch",
            "numpy_config": config.getvalue(),
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
