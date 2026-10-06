//! Experimental f64 kernels. Existing Python APIs remain the reference implementation.
use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;
use wide::f64x4;

type PyState<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);
type PyGhBatch<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
type NativeStep = (DVector<f64>, DMatrix<f64>, DVector<f64>, DMatrix<f64>);
type PyTrajectory<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
);

fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("BAYESIAN_FILTERS_RUST_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 32);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("Rust worker pool")
    })
}

// Owned column-major matrices with checked inner dimensions. GEMM dispatches SIMD
// at runtime; distributed wheels do not assume the build machine's CPU features.
fn multiply(a: &DMatrix<f64>, b: &DMatrix<f64>, ta: bool, tb: bool) -> DMatrix<f64> {
    let (m, k) = if ta {
        (a.ncols(), a.nrows())
    } else {
        a.shape()
    };
    let (bk, n) = if tb {
        (b.ncols(), b.nrows())
    } else {
        b.shape()
    };
    assert_eq!(k, bk);
    if m * n * k < 4096 {
        return match (ta, tb) {
            (false, false) => a * b,
            (true, false) => a.transpose() * b,
            (false, true) => a * b.transpose(),
            (true, true) => a.transpose() * b.transpose(),
        };
    }
    let mut out = DMatrix::zeros(m, n);
    let threads = pool().current_num_threads();
    let parallel = if threads > 1 && m * n * k >= 1_000_000 {
        gemm::Parallelism::Rayon(threads)
    } else {
        gemm::Parallelism::None
    };
    let calculate = |out: &mut DMatrix<f64>| {
        // SAFETY: dimensions are checked above; each owned matrix is contiguous
        // column-major, transposition swaps validated strides. Output never aliases
        // either input and remains alive for the entire synchronous GEMM call.
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                out.as_mut_ptr(),
                m as isize,
                1,
                false,
                a.as_ptr(),
                if ta { 1 } else { a.nrows() as isize },
                if ta { a.nrows() as isize } else { 1 },
                b.as_ptr(),
                if tb { 1 } else { b.nrows() as isize },
                if tb { b.nrows() as isize } else { 1 },
                0.0,
                1.0,
                false,
                false,
                false,
                parallel,
            );
        }
    };
    if threads > 1 && m * n * k >= 1_000_000 {
        pool().install(|| calculate(&mut out));
    } else {
        calculate(&mut out);
    }
    out
}

fn axpy(out: &mut [f64], input: &[f64], weight: f64) {
    let w = f64x4::splat(weight);
    let aligned = out.len() / 4 * 4;
    for (o, x) in out[..aligned]
        .chunks_exact_mut(4)
        .zip(input[..aligned].chunks_exact(4))
    {
        let v = f64x4::new(o.try_into().unwrap()) + f64x4::new(x.try_into().unwrap()) * w;
        o.copy_from_slice(&v.to_array());
    }
    for (o, x) in out[aligned..].iter_mut().zip(&input[aligned..]) {
        *o += x * weight;
    }
}

fn product(out: &mut [f64], a: &[f64], b: &[f64]) -> Result<f64, &'static str> {
    let mut sum = f64x4::ZERO;
    let aligned = out.len() / 4 * 4;
    for ((o, a), b) in out[..aligned]
        .chunks_exact_mut(4)
        .zip(a[..aligned].chunks_exact(4))
        .zip(b[..aligned].chunks_exact(4))
    {
        let a = f64x4::new(a.try_into().unwrap());
        let b = f64x4::new(b.try_into().unwrap());
        if !(a.is_finite() & b.is_finite() & a.simd_ge(f64x4::ZERO) & b.simd_ge(f64x4::ZERO)).all()
        {
            return Err("invalid probability");
        }
        let v = a * b;
        sum += v;
        o.copy_from_slice(&v.to_array());
    }
    let mut total = sum.reduce_add();
    for ((o, a), b) in out[aligned..]
        .iter_mut()
        .zip(&a[aligned..])
        .zip(&b[aligned..])
    {
        if !a.is_finite() || !b.is_finite() || *a < 0. || *b < 0. {
            return Err("invalid probability");
        }
        *o = a * b;
        total += *o;
    }
    Ok(total)
}

fn scale(out: &mut [f64], value: f64) {
    let aligned = out.len() / 4 * 4;
    let v = f64x4::splat(value);
    for chunk in out[..aligned].chunks_exact_mut(4) {
        let scaled = f64x4::new(chunk.try_into().unwrap()) * v;
        chunk.copy_from_slice(&scaled.to_array());
    }
    for x in &mut out[aligned..] {
        *x *= value;
    }
}

fn error(message: &str) -> PyErr {
    PyValueError::new_err(message.to_string())
}
fn matrix(a: &PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let v = a.as_array();
    DMatrix::from_row_iterator(v.nrows(), v.ncols(), v.iter().copied())
}
fn vector(a: &PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    DVector::from_iterator(a.len().unwrap(), a.as_array().iter().copied())
}
fn array2(a: &DMatrix<f64>) -> Array2<f64> {
    Array2::from_shape_fn((a.nrows(), a.ncols()), |(i, j)| a[(i, j)])
}
fn finite<'a>(values: impl Iterator<Item = &'a f64>) -> PyResult<()> {
    if values.into_iter().any(|v| !v.is_finite()) {
        return Err(error("inputs must be finite"));
    }
    Ok(())
}

#[pyfunction]
fn unscented_transform<'py>(
    py: Python<'py>,
    sigmas: PyReadonlyArray2<'py, f64>,
    wm: PyReadonlyArray1<'py, f64>,
    wc: PyReadonlyArray1<'py, f64>,
    noise: PyReadonlyArray2<'py, f64>,
) -> PyResult<PyState<'py>> {
    let s = sigmas.as_array();
    let wm = wm.as_array();
    let wc = wc.as_array();
    let q = noise.as_array();
    let (k, n) = s.dim();
    if n == 0 || k == 0 || wm.len() != k || wc.len() != k || q.dim() != (n, n) {
        return Err(error("incompatible sigma, weight or covariance dimensions"));
    }
    finite(s.iter().chain(wm.iter()).chain(wc.iter()).chain(q.iter()))?;
    let owned;
    let flat = if let Some(slice) = s.as_slice() {
        slice
    } else {
        owned = s.iter().copied().collect::<Vec<_>>();
        &owned
    };
    let mut mean = vec![0.; n];
    for (i, row) in flat.chunks_exact(n).enumerate() {
        axpy(&mut mean, row, wm[i]);
    }
    let deviations = DMatrix::from_fn(k, n, |i, j| s[(i, j)] - mean[j]);
    let mut weighted = deviations.clone();
    for j in 0..n {
        for i in 0..k {
            weighted[(i, j)] *= wc[i];
        }
    }
    let mut cov = multiply(&deviations, &weighted, true, false);
    for i in 0..n {
        for j in 0..n {
            cov[(i, j)] += q[(i, j)];
        }
    }
    Ok((mean.into_pyarray(py), array2(&cov).into_pyarray(py)))
}

#[pyfunction]
fn gh_batch<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    mut x: f64,
    mut dx: f64,
    dt: f64,
    g: f64,
    h: f64,
) -> PyResult<PyGhBatch<'py>> {
    if dt == 0. || ![x, dx, dt, g, h].iter().all(|v| v.is_finite()) {
        return Err(error("finite parameters and nonzero dt required"));
    }
    let data = data.as_array();
    finite(data.iter())?;
    let mut out = Array2::zeros((data.len() + 1, 2));
    let mut predictions = Vec::with_capacity(data.len());
    out[(0, 0)] = x;
    out[(0, 1)] = dx;
    for (i, z) in data.iter().enumerate() {
        let pred = x + dx * dt;
        let residual = z - pred;
        dx += h / dt * residual;
        x = pred + g * residual;
        predictions.push(pred);
        out[(i + 1, 0)] = x;
        out[(i + 1, 1)] = dx;
    }
    Ok((out.into_pyarray(py), predictions.into_pyarray(py)))
}

#[pyfunction]
fn discrete_update<'py>(
    py: Python<'py>,
    likelihood: PyReadonlyArray1<'py, f64>,
    prior: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = likelihood.as_array();
    let b = prior.as_array();
    if a.is_empty() || a.len() != b.len() {
        return Err(error("nonempty equal-length distributions required"));
    }
    let a_owned;
    let b_owned;
    let a = if let Some(v) = a.as_slice() {
        v
    } else {
        a_owned = a.iter().copied().collect::<Vec<_>>();
        &a_owned
    };
    let b = if let Some(v) = b.as_slice() {
        v
    } else {
        b_owned = b.iter().copied().collect::<Vec<_>>();
        &b_owned
    };
    let mut out = vec![0.; a.len()];
    let parallel = a.len() >= 131072 && pool().current_num_threads() > 1;
    let sum = if parallel {
        pool()
            .install(|| {
                out.par_chunks_mut(32768)
                    .zip(a.par_chunks(32768))
                    .zip(b.par_chunks(32768))
                    .map(|((o, a), b)| product(o, a, b))
                    .collect::<Result<Vec<f64>, _>>()
            })
            .map_err(error)?
            .iter()
            .sum()
    } else {
        product(&mut out, a, b).map_err(error)?
    };
    if !sum.is_finite() || sum <= 0. {
        return Err(error("posterior mass must be positive and finite"));
    }
    if !(1. / sum).is_finite() {
        // Dividing by a subnormal sum can be finite even when its reciprocal is not.
        for value in &mut out {
            *value /= sum;
        }
    } else if parallel {
        pool().install(|| {
            out.par_chunks_mut(32768)
                .for_each(|chunk| scale(chunk, 1. / sum))
        });
    } else {
        scale(&mut out, 1. / sum);
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn discrete_predict<'py>(
    py: Python<'py>,
    pdf: PyReadonlyArray1<'py, f64>,
    offset: i64,
    kernel: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pdf.as_array();
    let k = kernel.as_array();
    let n = p.len();
    if n == 0 || k.is_empty() {
        return Err(error("nonempty pdf and kernel required"));
    }
    finite(p.iter().chain(k.iter()))?;
    let p_owned;
    let p = if let Some(v) = p.as_slice() {
        v
    } else {
        p_owned = p.iter().copied().collect::<Vec<_>>();
        &p_owned
    };
    let mut out = vec![0.; n];
    let shift = offset.rem_euclid(n as i64);
    let center = (k.len() / 2) as i64;
    let fill = |out: &mut [f64], start: usize| {
        for (j, w) in k.iter().enumerate() {
            let source = (start as i64 + center - j as i64 - shift).rem_euclid(n as i64) as usize;
            let first = (n - source).min(out.len());
            axpy(&mut out[..first], &p[source..source + first], *w);
            if first < out.len() {
                let remaining = out.len() - first;
                axpy(&mut out[first..], &p[..remaining], *w);
            }
        }
    };
    if n >= 131072 && pool().current_num_threads() > 1 {
        pool().install(|| {
            out.par_chunks_mut(32768)
                .enumerate()
                .for_each(|(i, chunk)| fill(chunk, i * 32768))
        });
    } else {
        fill(&mut out, 0);
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn resample<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let w = weights.as_array();
    let p = positions.as_array();
    if w.is_empty() {
        return Err(error("weights must not be empty"));
    }
    finite(w.iter().chain(p.iter()))?;
    if w.iter().any(|v| *v < 0.) || p.iter().any(|v| *v < 0. || *v >= 1.) {
        return Err(error("invalid weights or positions"));
    }
    let sum: f64 = w.iter().sum();
    if (sum - 1.).abs() > 1e-10 {
        return Err(error("weights must sum to one"));
    }
    if p.iter().zip(p.iter().skip(1)).any(|(a, b)| b < a) {
        return Err(error("positions must be sorted"));
    }
    let mut j = 0;
    let mut cumulative = w[0];
    let mut out = Vec::with_capacity(p.len());
    for value in p {
        while *value >= cumulative && j + 1 < w.len() {
            j += 1;
            cumulative += w[j];
        }
        out.push(j as i64);
    }
    Ok(out.into_pyarray(py))
}

struct LinearModel {
    f: DMatrix<f64>,
    q: DMatrix<f64>,
    h: DMatrix<f64>,
    r: DMatrix<f64>,
}
fn model(
    x: &DVector<f64>,
    p: &DMatrix<f64>,
    f: DMatrix<f64>,
    q: DMatrix<f64>,
    h: DMatrix<f64>,
    r: DMatrix<f64>,
    m: usize,
) -> PyResult<LinearModel> {
    let n = x.len();
    if n == 0
        || m == 0
        || p.shape() != (n, n)
        || f.shape() != (n, n)
        || q.shape() != (n, n)
        || h.shape() != (m, n)
        || r.shape() != (m, m)
    {
        return Err(error("incompatible Kalman dimensions"));
    }
    finite(
        x.iter()
            .chain(p.iter())
            .chain(f.iter())
            .chain(q.iter())
            .chain(h.iter())
            .chain(r.iter()),
    )?;
    Ok(LinearModel { f, q, h, r })
}
fn step(
    x: &DVector<f64>,
    p: &DMatrix<f64>,
    z: &DVector<f64>,
    model: &LinearModel,
) -> PyResult<NativeStep> {
    let xp = &model.f * x;
    let pp = multiply(&multiply(&model.f, p, false, false), &model.f, false, true) + &model.q;
    let residual = z - &model.h * &xp;
    let pht = multiply(&pp, &model.h, false, true);
    let s = multiply(&model.h, &pht, false, false) + &model.r;
    let inverse = s
        .try_inverse()
        .ok_or_else(|| error("singular innovation covariance"))?;
    let gain = multiply(&pht, &inverse, false, false);
    let xnew = &xp + &gain * residual;
    // Match the reference's Joseph covariance update, not the less stable simplified form.
    let ikh = DMatrix::identity(x.len(), x.len()) - multiply(&gain, &model.h, false, false);
    let pnew = multiply(&multiply(&ikh, &pp, false, false), &ikh, false, true)
        + multiply(&multiply(&gain, &model.r, false, false), &gain, false, true);
    Ok((xnew, pnew, xp, pp))
}

// Explicit matrices match the public numerical kernel contract.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn kalman_step<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    f: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray2<'py, f64>,
    r: PyReadonlyArray2<'py, f64>,
) -> PyResult<PyState<'py>> {
    let x = vector(&x);
    let p = matrix(&p);
    let z = vector(&z);
    finite(z.iter())?;
    let model = model(
        &x,
        &p,
        matrix(&f),
        matrix(&q),
        matrix(&h),
        matrix(&r),
        z.len(),
    )?;
    let (x, p, _, _) = step(&x, &p, &z, &model)?;
    Ok((
        x.as_slice().to_vec().into_pyarray(py),
        array2(&p).into_pyarray(py),
    ))
}

// Explicit matrices match the public numerical kernel contract.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn kalman_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    zs: PyReadonlyArray2<'py, f64>,
    f: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray2<'py, f64>,
    r: PyReadonlyArray2<'py, f64>,
) -> PyResult<PyTrajectory<'py>> {
    let mut x = vector(&x);
    let mut p = matrix(&p);
    let zs = zs.as_array();
    finite(zs.iter())?;
    let (t, m) = zs.dim();
    let n = x.len();
    let model = model(&x, &p, matrix(&f), matrix(&q), matrix(&h), matrix(&r), m)?;
    let mut xs = Array2::zeros((t, n));
    let mut ps = Array3::zeros((t, n, n));
    let mut xp = Array2::zeros((t, n));
    let mut pp = Array3::zeros((t, n, n));
    for time in 0..t {
        let z = DVector::from_iterator(m, zs.row(time).iter().copied());
        let (xn, pn, xprior, pprior) = step(&x, &p, &z, &model)?;
        for i in 0..n {
            xs[(time, i)] = xn[i];
            xp[(time, i)] = xprior[i];
            for j in 0..n {
                ps[(time, i, j)] = pn[(i, j)];
                pp[(time, i, j)] = pprior[(i, j)];
            }
        }
        x = xn;
        p = pn;
    }
    Ok((
        xs.into_pyarray(py),
        ps.into_pyarray(py),
        xp.into_pyarray(py),
        pp.into_pyarray(py),
    ))
}

#[pymodule]
fn _bayesian_filters_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unscented_transform, m)?)?;
    m.add_function(wrap_pyfunction!(gh_batch, m)?)?;
    m.add_function(wrap_pyfunction!(discrete_update, m)?)?;
    m.add_function(wrap_pyfunction!(discrete_predict, m)?)?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_step, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_batch, m)?)?;
    Ok(())
}
