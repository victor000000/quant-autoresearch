#!/usr/bin/env python3
"""Check whether stripping comments+docstrings gets the rendered QC script under
QC's 64,000-char main.py limit. Stdlib only (no numpy needed)."""
import sys, ast
from lb.harness.orchestrator import render_script

def strip(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
               and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.unparse(tree)

for ax in ["vol", "dollar", "tick", "entropy"]:
    src = render_script("GLD", axis=ax)
    mini = strip(src)
    compile(mini, "<mini>", "exec")  # must stay valid
    print(f"axis={ax:9s} raw={len(src):6d}  minified={len(mini):6d}  under_64k={len(mini) < 64000}")
