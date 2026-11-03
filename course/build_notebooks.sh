#!/bin/sh
# Convert the percent-format sources in notebooks/src/*.py into notebooks/*.ipynb (no outputs).
# Usage: sh build_notebooks.sh [--execute]   (--execute also runs each notebook top to bottom as a test)
set -e
cd "$(dirname "$0")"
for f in notebooks/src/*.py; do
  n=notebooks/$(basename "${f%.py}").ipynb
  uv run --group dev jupytext --quiet --to notebook --output "$n" "$f"
  if [ "$1" = "--execute" ]; then
    uv run --group dev jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
      --output-dir "${OUT:-.cache/executed}" "$n"
  fi
done
