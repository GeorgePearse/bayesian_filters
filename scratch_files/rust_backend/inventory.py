"""Record the full Python surface so a native prototype is not mistaken for a port."""

import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
records = []
for path in sorted((root / "bayesian_filters").rglob("*.py")):
    if any(p in {"tests", "examples", "testing_utils", "__pycache__"} for p in path.parts) or path.name in {
        "__init__.py",
        "rust.py",
        "backends.py",
    }:
        continue
    tree = ast.parse(path.read_text())
    entries = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            entries.append(
                {
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "methods": [
                        m.name for m in node.body if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")
                    ]
                    if isinstance(node, ast.ClassDef)
                    else [],
                }
            )
    relative = str(path.relative_to(root))
    status = "not ported; evaluate separately"
    if relative.endswith("unscented_transform.py"):
        status = "native default weighted moments; custom callbacks use Python reference"
    elif relative.endswith("kalman_filter.py"):
        status = "restricted native predict/update and fixed-matrix batch; full class API not ported"
    elif relative.endswith("gh_filter.py"):
        status = "native scalar GH batch only; GHK/order variants and stateful methods not ported"
    elif relative.endswith("resampling.py"):
        status = "native systematic/stratified traversal; NumPy RNG; residual/multinomial remain reference"
    elif relative.endswith("discrete_bayes.py"):
        status = "native 1D update and integer-offset wrap prediction; normalize/other modes remain reference"
    elif relative.endswith("stats.py"):
        status = "not ported; plotting remains Python by design; numerical statistics unmeasured"
    records.append({"module": relative, "status": status, "public_definitions": entries})
(root / "scratch_files/rust_backend/inventory.json").write_text(json.dumps(records, indent=2) + "\n")
print(len(records), "modules", sum(len(x["public_definitions"]) for x in records), "top-level public definitions")
