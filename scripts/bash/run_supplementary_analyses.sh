#!/usr/bin/env bash
# Reproduce supplementary analyses from the EMSE paper (Llama failures, GPT-4o cost, GHA features).
# Run from ReplicationPackage root after activating the virtual environment.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXTRACTED="$ROOT/results/generated_outputs/extracted/CIgrate_New_Results-OurCleanData-FinalHyperparameters"

if [[ ! -d "$EXTRACTED/CIgrate_Results_FewShot" ]]; then
  echo "Extracting bundled YAML outputs (required for Llama failure + GHA feature analyses)..."
  mkdir -p "$ROOT/results/generated_outputs/extracted"
  unzip -q -o "$ROOT/results/generated_outputs/CIgrate_New_Results-OurCleanData-FinalHyperparameters.zip" \
    -d "$ROOT/results/generated_outputs/extracted"
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

echo "==> Llama failure-mode classification (RQ1)"
"$PYTHON" scripts/analysis/analyze_llama_failures.py

echo "==> GHA workflow feature taxonomy (RQ2 outlier analysis)"
"$PYTHON" scripts/analysis/gha_feature_analysis.py

if [[ -d "$ROOT/../CIgrate_New_Results-OurCleanData-GPT-4o" ]]; then
  echo "==> GPT-4o token/cost estimate (Discussion)"
  "$PYTHON" scripts/analysis/estimate_gpt4o_cost.py
else
  echo "==> Skipping GPT-4o cost recompute (precomputed CSV in results/analysis/gpt4o_cost_estimate.csv)"
  echo "    To recompute, place GPT-4o outputs at ../CIgrate_New_Results-OurCleanData-GPT-4o"
fi

echo "Done. Outputs in results/analysis/"
