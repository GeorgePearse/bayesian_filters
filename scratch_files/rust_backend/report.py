"""Generate a self-contained interactive report and a standard SVG benchmark plot."""

import html
import io
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
single = json.loads((root / "results.json").read_text())
parallel = json.loads((root / "results_parallel.json").read_text())
initial = json.loads((root / "results_initial.json").read_text())
prior = {r["case"]: r for r in initial["results"]}
rows = []
for report in [single, parallel]:
    threads = report["method"]["threads"]
    for result in report["results"]:
        row = dict(result, threads=threads)
        before = prior.get(row["case"])
        row["rust_improvement"] = (
            before["microseconds"]["rust"] / row["microseconds"]["rust"] if before and threads == 1 else None
        )
        rows.append(row)
fig, ax = plt.subplots(figsize=(12, 12))
labels = [x["case"] for x in single["results"]]
speeds = [x["speedup"] for x in single["results"]]
ax.barh(labels, speeds, color=["#168566" if x >= 1 else "#b34b42" for x in speeds])
ax.set_xscale("log")
ax.axvline(1, color="#444", linewidth=1)
ax.invert_yaxis()
ax.set_xlabel("Reference time / Rust time (log scale; above 1 means Rust is faster)")
ax.set_title("Optional Rust backend — one thread, complete Python calls")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
svg = io.StringIO()
fig.savefig(svg, format="svg")
plt.close(fig)
svg_text = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
(root / "speedups.svg").write_text(svg_text)
metadata = html.escape(json.dumps({"method": single["method"], "environment": single["environment"]}, indent=2))
page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Bayesian Filters: optional Rust backend benchmarks</title>
<style>body{font:16px system-ui;max-width:1250px;margin:3rem auto;padding:0 1rem;color:#18232f;background:#fafbfc}h1{font-size:2rem}p{line-height:1.65;max-width:90ch}input,select{font:inherit;padding:.5rem}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:.6rem;text-align:right;border-bottom:1px solid #d5dce3}td:first-child,th:first-child{text-align:left}th{cursor:pointer;background:#edf1f5}section{overflow:auto}svg{max-width:100%;height:auto}.fast{color:#086b4d}.slow{color:#a22a27}pre{white-space:pre-wrap}small{color:#46576a}</style>
<h1>Bayesian Filters: optional Rust backend</h1>
<p>The existing NumPy/SciPy code and defaults are preserved. This is a measured native-kernel prototype, <strong>not a full native port</strong>. Each timed case first passes numerical agreement checks. Timings include the Python boundary, validation and allocations. Higher speedup is better; values below 1 expose a Rust slowdown.</p>
<p>Release build; nine alternating paired rounds; fixed inputs and CPU affinity. Compare the one-thread and four-thread runs separately: each uses matching NumPy BLAS and Rust thread limits. SIMD/GEMM and Rayon work thresholds mean small kernels do not necessarily start parallel work. Vectorized NumPy controls identify gains available without changing languages. These are measurements on one shared development host.</p>
<label>Threads <select id="threads"><option value="1">1</option><option value="4">4</option></select></label>
<label>Filter <input id="filter" placeholder="Kalman, UKF, resample…"></label>
<p><small>Click a column heading to sort. Times are microseconds per complete call. Initial comparison uses the recorded pre-optimization single-thread prototype.</small></p>
<section><table><thead><tr><th data-key="case">Workload</th><th data-key="numpy">NumPy µs</th><th data-key="rust">Rust µs</th><th data-key="speedup">Speedup</th><th data-key="control">Vectorized µs</th><th data-key="optimized">Vs vectorized</th><th data-key="improvement">Rust improvement</th><th data-key="error">Max error</th></tr></thead><tbody></tbody></table></section>
<h2>Single-thread comparison</h2>CHART
<h2>Reproduce and interpret</h2><p>See <code>scratch_files/RUST_BACKEND.md</code>, <code>benchmark.py</code>, the raw JSON files and the API inventory in this PR. Timings are not correctness proofs for unported APIs. Custom unscented mean/residual callbacks use the unchanged reference implementation. Full UKF timings include existing Python callbacks and sigma-point work.</p>
<details><summary>Environment and method</summary><pre>METADATA</pre></details>
<script>const rows=DATA;let key='case',ascending=true;
function value(r,k){return ({numpy:r.microseconds.numpy,rust:r.microseconds.rust,control:r.microseconds.numpy_vectorized,optimized:r.speedup_vs_vectorized,improvement:r.rust_improvement,error:r.max_abs_error})[k]??r[k]??null}
function render(){const query=document.querySelector('#filter').value.toLowerCase(),threads=Number(document.querySelector('#threads').value);const selected=rows.filter(r=>r.threads===threads&&r.case.toLowerCase().includes(query));selected.sort((a,b)=>{const x=value(a,key),y=value(b,key);return (typeof x==='string'?x.localeCompare(y):(x??-Infinity)-(y??-Infinity))*(ascending?1:-1)});document.querySelector('tbody').replaceChildren(...selected.map(r=>{const tr=document.createElement('tr');for(const k of ['case','numpy','rust','speedup','control','optimized','improvement','error']){const td=document.createElement('td'),v=value(r,k);td.textContent=v==null?'—':typeof v==='string'?v:k==='error'?v.toExponential(2):v.toFixed(2)+(['speedup','optimized','improvement'].includes(k)?'×':'');if(['speedup','optimized','improvement'].includes(k)&&v!=null)td.className=v>=1?'fast':'slow';tr.append(td)}return tr}))}
document.querySelector('#threads').onchange=render;document.querySelector('#filter').oninput=render;document.querySelectorAll('th').forEach(th=>th.onclick=()=>{ascending=key===th.dataset.key?!ascending:true;key=th.dataset.key;render()});render();</script></html>"""
page = (
    page.replace("CHART", svg_text.split("?>", 1)[-1])
    .replace("METADATA", metadata)
    .replace("DATA", json.dumps(rows).replace("</", "<\\/"))
)
(root / "report.html").write_text(page.rstrip() + "\n")
print(root / "report.html")

summary = [
    "# Measured backend comparisons",
    "",
    "Release build on " + single["environment"]["cpu"] + "; Python " + single["environment"]["python"] + ".",
    "Nine paired rounds; times include the Python boundary. Existing NumPy/SciPy code remains the default.",
    "Rust is faster above 1x. Thread counts are matched between BLAS and Rust within each run.",
    "",
    "| Workload | 1-thread speedup | 4-thread speedup |",
    "| --- | ---: | ---: |",
]
parallel_by_case = {r["case"]: r for r in parallel["results"]}
for result in single["results"]:
    other = parallel_by_case[result["case"]]
    summary.append(f"| {result['case']} | {result['speedup']:.2f}x | {other['speedup']:.2f}x |")
summary += [
    "",
    "Large dense Kalman steps still favor the existing BLAS path on this host despite SIMD/GEMM and Rayon. "
    "Vectorized NumPy controls can also beat native kernels; see the interactive report for that comparison. "
    "The optional backend is a measured subset, not a full native port.",
    "",
    f"Maximum absolute discrepancy across these cases: {max(r['max_abs_error'] for r in rows):.3g}; "
    "all cases pass rtol=atol=2e-9 before timing. This does not establish parity for unported APIs.",
    "",
    "[Interactive report](report.html) · [Raw single-thread results](results.json) · "
    "[Raw parallel results](results_parallel.json) · [API inventory](inventory.json)",
]
(root / "summary.md").write_text("\n".join(summary) + "\n")
