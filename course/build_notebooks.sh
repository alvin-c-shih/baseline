#!/bin/sh
# Convert the percent-format sources in notebooks/src/*.py into notebooks/*.ipynb.
# Usage: sh build_notebooks.sh            update the notebooks from their sources, keeping the outputs of every code cell
#                                         whose code is unchanged: enough after editing the prose
#        sh build_notebooks.sh --execute  rebuild and run each one top to bottom, refreshing all outputs in notebooks/*.ipynb
#                                         (the committed copies). The tier defaults to laptop: BASELINE_TIER=… to change it.
set -e
cd "$(dirname "$0")"
for f in notebooks/src/*.py; do
  n=notebooks/$(basename "${f%.py}").ipynb
  if [ "$1" = "--execute" ]; then
    uv run --group dev jupytext --quiet --to notebook --output "$n" "$f"
    BASELINE_TIER="${BASELINE_TIER:-laptop}" HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error \
      uv run --group dev jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$n"
  else
    upd=; [ -f "$n" ] && upd=--update
    uv run --group dev jupytext --quiet $upd --to notebook --output "$n" "$f"
  fi
done
