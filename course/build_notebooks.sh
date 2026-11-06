#!/bin/sh
# Convert the percent-format sources in notebooks/src/*.py into notebooks/*.ipynb.
# Usage: sh build_notebooks.sh            notebooks without outputs
#        sh build_notebooks.sh --execute  also run each one top to bottom and keep the outputs in notebooks/*.ipynb
#                                         (the committed copies). The tier defaults to laptop: BASELINE_TIER=… to change it.
set -e
cd "$(dirname "$0")"
for f in notebooks/src/*.py; do
  n=notebooks/$(basename "${f%.py}").ipynb
  uv run --group dev jupytext --quiet --to notebook --output "$n" "$f"
  if [ "$1" = "--execute" ]; then
    BASELINE_TIER="${BASELINE_TIER:-laptop}" HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error \
      uv run --group dev jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$n"
  fi
done
