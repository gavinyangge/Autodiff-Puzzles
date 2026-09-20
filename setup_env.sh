#!/usr/bin/env bash
# Reproducible setup for Autodiff-Puzzles (local, CPU-only JAX — no GPU needed).
# Mirror of how the `tensor-puzzles` env is set up, but for JAX + chalx.
#
#   bash setup_env.sh
#
set -euo pipefail

ENV=autodiff-puzzles
ENVPREFIX=/opt/homebrew/Caskroom/miniconda/base/envs/$ENV
PYBIN=$ENVPREFIX/bin/python

echo "==> 1. create conda env ($ENV, python 3.12)"
conda create -n "$ENV" python=3.12 -y

echo "==> 2. install deps via conda-forge"
# NOTE: `pip install jax` hangs on dependency backtracking; conda-forge resolves fast.
conda install -n "$ENV" -c conda-forge -y --solver=libmamba \
  jax jaxlib numpy matplotlib jaxtyping beartype ipykernel jupyter debugpy \
  chex toolz colour imageio typing-extensions importlib_metadata

echo "==> 3. install chalx (JAX-compatible fork of chalk-diagrams, imports as 'chalk')"
$PYBIN -m pip install --no-input --no-deps git+https://github.com/chalk-diagrams/chalx

echo "==> 4. patch chalx for modern jax (>=0.5 removed jax.tree_map, which chalx 0.2.1 uses)"
$PYBIN - <<'PY'
import os, chalk
f = os.path.join(os.path.dirname(chalk.__file__), "array_types.py")
s = open(f).read()
s2 = s.replace("jax.tree_map(", "jax.tree.map(")
if s != s2:
    open(f, "w").write(s2)
    print("patched chalx array_types.py: jax.tree_map -> jax.tree.map")
else:
    print("chalx already patched (or pattern not found) — nothing to do")
PY

echo "==> 5. register jupyter kernel for VSCode (with debugger enabled)"
$PYBIN -m ipykernel install --user --name "$ENV" --display-name "Python ($ENV)"

echo
echo "Done. In VSCode open autodiff_puzzlers.ipynb and select the kernel"
echo "  'Python (autodiff-puzzles)'."
echo "Cell 1's !pip install line is commented out (deps already in the env)."
