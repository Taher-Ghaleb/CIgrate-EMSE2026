#!/usr/bin/env bash
# Run CIgrate open-weight model experiments via Unsloth.
#
# GPT-4o is run separately: scripts/prompting/CIgrate_gpt-4o.py
#
# Usage (from ReplicationPackage root, GPU + Unsloth env required):
#
#   export MODELS_PATH=/path/to/UNSLOTH_MODELS   # HuggingFace model cache root
#   bash scripts/orchestration/run_cigrate_experiments.sh rq1
#   bash scripts/orchestration/run_cigrate_experiments.sh rq2-cv
#   bash scripts/orchestration/run_cigrate_experiments.sh rq2-final
#   bash scripts/orchestration/run_cigrate_experiments.sh rq4
#   bash scripts/orchestration/run_cigrate_experiments.sh all
#
# Optional overrides:
#   OUTPUT_BASE=results/recomputed_outputs
#   LOAD_4BIT=True
#   FOLD=03                        # for rq2-cv, run a single fold only (01-10)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
UNSLOTH_SCRIPT="$ROOT/scripts/fine_tuning/CIgrate_fine_tune_unsloth_with_final_hyperparameters.py"
CONFIG_PATH="$ROOT/data"
DATA_ROOT="$ROOT/data/CIgrate_FineTuning_Data/OurCleanData"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT/results/recomputed_outputs/OurCleanData-FinalHyperparameters}"

MODELS_PATH="${MODELS_PATH:?Set MODELS_PATH to your local HuggingFace/Unsloth model directory}"
LOAD_4BIT="${LOAD_4BIT:-True}"

run_job() {
  local fold_set="$1"      # e.g. Sample_153/Set_Full
  local model="$2"           # HuggingFace model id
  local migration="$3"       # travis_to_gha | gha_to_travis
  local mode="$4"            # finetune | zero_shot | few_shot | just_generate
  local max_seq="${5:-4096}"

  local results_folder
  case "$mode" in
    finetune) results_folder="CIgrate_Results_FineTuned" ;;
    zero_shot) results_folder="CIgrate_Results_ZeroShot" ;;
    few_shot) results_folder="CIgrate_Results_FewShot" ;;
    just_generate) results_folder="CIgrate_Results_PullRequests" ;;
    *) echo "Unknown mode: $mode"; exit 1 ;;
  esac

  local input_path="$DATA_ROOT/$fold_set"
  local output_path="$OUTPUT_BASE/$results_folder/$fold_set"
  if [[ "$mode" == "just_generate" ]]; then
    input_path="$input_path/$migration"
    output_path="$output_path/$migration"
  fi

  mkdir -p "$output_path"
  echo ">>> $mode | $model | $migration | $fold_set"
  "$PYTHON" -u "$UNSLOTH_SCRIPT" \
    "$MODELS_PATH" "$model" "$max_seq" "$LOAD_4BIT" \
    "$CONFIG_PATH" "$input_path" "$output_path" "$migration" "$mode"
}

run_rq1() {
  local models=(
    "google/gemma-3-12b-it"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
  )
  local modes=(zero_shot few_shot)
  for model in "${models[@]}"; do
    local max_seq=4096
    [[ "$model" == mistralai/* ]] && max_seq=8192
    for mode in "${modes[@]}"; do
      for migration in travis_to_gha gha_to_travis; do
        run_job "Sample_153/Set_Full" "$model" "$migration" "$mode" "$max_seq"
      done
    done
  done
}

run_rq2_cv() {
  local folds=(01 02 03 04 05 06 07 08 09 10)
  if [[ -n "${FOLD:-}" ]]; then folds=("$FOLD"); fi
  for f in "${folds[@]}"; do
    for migration in travis_to_gha gha_to_travis; do
      run_job "Sample_153/Set_90_10/Set_90_10_V_${f}" "google/gemma-3-12b-it" "$migration" finetune
    done
  done
}

run_rq2_final() {
  for migration in travis_to_gha gha_to_travis; do
    run_job "Sample_153/Set_100_0" "google/gemma-3-12b-it" "$migration" finetune
  done
}

run_rq4() {
  for migration in travis_to_gha gha_to_travis; do
    run_job "Sample_30_30/Set_30_30" "google/gemma-3-12b-it" "$migration" just_generate
  done
}

case "${1:-}" in
  rq1) run_rq1 ;;
  rq2-cv) run_rq2_cv ;;
  rq2-final) run_rq2_final ;;
  rq4) run_rq4 ;;
  all)
    run_rq1
    run_rq2_cv
    run_rq2_final
    run_rq4
    ;;
  *)
    echo "Usage: $0 {rq1|rq2-cv|rq2-final|rq4|all}"
    exit 1
    ;;
esac

echo "Done. Outputs under $OUTPUT_BASE"
