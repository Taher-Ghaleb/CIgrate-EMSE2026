# CIgrate Replication Package

Replication package for the paper **"CIgrate: Automating CI Service Migration with Large Language Models"** (submitted to EMSE). 

This repository provides the data, scripts, precomputed results, and analysis tooling needed to reproduce the experiments and findings reported in the paper.

## Package Structure

```
ReplicationPackage/
├── data/
│   ├── prompts/                         # System and user prompt templates
│   ├── few-shot-examples/               # Static few-shot migration examples
│   ├── CIgrate_FineTuning_Data/
│   │   └── OurCleanData/                # Curated ground-truth instruction CSVs
│   │       ├── Sample_153/              # Primary evaluation (153 projects)
│   │       │   ├── Set_Full/            # Zero-shot & few-shot prompting (RQ1)
│   │       │   ├── Set_90_10/           # 10-fold CV for fine-tuning (RQ2)
│   │       │   └── Set_100_0/           # Final models for PR generation (RQ4)
│   │       └── Sample_30_30/            # Pull-request study projects (RQ4)
│   │           └── Set_30_30/
│   ├── supplementary/                   # PR spreadsheet, sample migration pairs
│   └── OurCleanData_153_projects.csv
├── scripts/
│   ├── fine_tuning/                       # Unsloth: fine-tune + zero/few-shot + PR gen
│   ├── proprietary/                         # GPT-4o API prompting only
│   ├── analysis/                          # Metrics, figures, linting, supplements
│   └── bash/                     # Environment setup and experiment runners
├── results/
│   ├── figures/                           # All paper figures (overview + results)
│   ├── tables/                            # Pre-generated LaTeX tables
│   ├── statistics/                        # Wilcoxon, Mann-Whitney, linting, RQ4 tests
│   ├── similarity/                        # Precomputed Cosine + CrystalBLEU scores
│   ├── linting/                           # Per-file lint results + aggregations
│   ├── analysis/                          # Llama failures, GPT cost, GHA features
│   ├── generated_outputs/                 # Generated YAML outputs (zip archives)
│   ├── rq4/                               # RQ4 PR selection and analysis
│   └── rq4_pull_requests/                 # PR study configs and summaries
├── requirements.txt
├── LICENSE
└── README.md
```

## Experiment Pipeline

All **open-weight models** (Gemma 3 12B, Llama 3.1 8B, Mistral 7B) are run through a **single Unsloth script**:

`scripts/fine_tuning/CIgrate_fine_tune_unsloth_with_final_hyperparameters.py`

| Mode | Purpose | Script `mode` argument |
|------|---------|------------------------|
| Fine-tuning | LoRA training + evaluation | `finetune` |
| Zero-shot | Prompt-only migration | `zero_shot` |
| Few-shot | 3-shot in-context migration | `few_shot` |
| PR generation | Inference with saved fine-tuned weights | `just_generate` |

**GPT-4o** is the only model run separately via `scripts/proprietary/CIgrate_gpt-4o.py` (OpenAI API).

### Dataset splits used in the paper

| Sample / Set | Used for |
|--------------|----------|
| `Sample_153/Set_Full` | RQ1 zero-shot & few-shot (all open-weight models) |
| `Sample_153/Set_90_10` (10 folds) | RQ2 fine-tuning cross-validation (Gemma 3 12B) |
| `Sample_153/Set_100_0` | Final fine-tuned models for RQ4 PR migrations |
| `Sample_30_30/Set_30_30` | RQ4 pull-request study inputs |

### CIMig baseline data and results

Raw CIMig YAML pairs are **not** included in this package. You may obtain the original 251-project data from the CIMig paper: [figshare](https://figshare.com/s/d903576fab38e2a54660).

Precomputed CIgrate outputs on the noisy 251-project data are included under `CIMig_Results/` inside the generated OurCleanData results tree (see below). Raw CIMig YAML pairs are available from the CIMig paper: [figshare](https://figshare.com/s/d903576fab38e2a54660).

## Research Questions Covered

| RQ | Question | Key artifacts |
|----|----------|---------------|
| **RQ1** | Zero-shot vs few-shot prompting | `results/similarity/`, `results/generated_outputs/`, `results/analysis/llama_failure_mode_*.csv` |
| **RQ2** | Fine-tuning improvement | `Set_90_10/` folds, `results/analysis/gha_feature_*.csv` |
| **RQ3** | CIgrate vs CIMig baseline | Similarity CSVs, linting, CIMig results zip |
| **RQ4** | Practical deployment (PR study) | `Sample_30_30/`, `results/rq4/`, `results/rq4_pull_requests/` |

## Figures

All paper figures are in `results/figures/`:

| File | Description |
|------|-------------|
| `study_overview.pdf` | Study design overview |
| `ci_workflow.pdf` | CI migration workflow |
| `Background_motivational_example.pdf` | Motivating example |
| `fig_similarity_comparison.pdf` | RQ1 similarity comparison |
| `fig_finetuning_folds_boxplot.pdf` | RQ2 fold-level box plots |
| `fig_boxplot_comparison.pdf` | RQ3 model comparison |
| `fig_improvement_heatmap.pdf` | RQ3 improvement heatmap |
| `fig_linting_results.pdf` | RQ3 linting results |
| `fig_rq4_*.pdf` | RQ4 PR study charts |

## Installation

Python **3.11+** and a **CUDA GPU** are required for open-weight model experiments (fine-tuning and Unsloth inference).

```bash
cd ReplicationPackage
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Unsloth on HPC clusters, see `scripts/bash/unsloth_env_setup.sh` (adapt paths locally).

### Additional requirements

| Component | Used for | Setup |
|-----------|----------|-------|
| **OpenAI API key** | GPT-4o (RQ1) | `export OPENAI_API_KEY=your_key_here` |
| **CUDA GPU** | Unsloth fine-tuning & inference | NVIDIA driver + PyTorch with CUDA |
| **HuggingFace model weights** | Open-weight models | Download to `MODELS_PATH` (see below) |
| **actionlint** | GitHub Actions linting (optional) | [github.com/rhysd/actionlint](https://github.com/rhysd/actionlint) |
| **travis-lint** | Travis CI linting (optional) | `gem install travis` |

### Model weights

Download base models to a local directory and set `MODELS_PATH`:

```bash
export MODELS_PATH=/path/to/UNSLOTH_MODELS
# Expected layout: $MODELS_PATH/google_gemma-3-12b-it/, etc.
```

LoRA adapters and full fine-tuned weights (~36 GB) are **not** included. Contact the authors or re-run fine-tuning (see below).

## Usage

### Step 0 — Use precomputed results (fastest)

| Artifact | Location |
|----------|----------|
| Similarity scores | `results/similarity/` |
| Paper figures | `results/figures/` |
| LaTeX tables | `results/tables/` |
| Statistical tests | `results/statistics/` |
| Linting (3,251 rows) | `results/linting/linting_detailed_results.csv` |
| All generated migrations | `results/generated_outputs/CIgrate_New_Results-OurCleanData-FinalHyperparameters/` — includes `CIgrate_Results_{ZeroShot,FewShot,FineTuned}` and `CIMig_Results/` |
| Supplementary analyses | `results/analysis/` |

Regenerate core figures:

```bash
python scripts/analysis/generate_paper_figures.py
```

Supplementary analyses (Llama failures, GPT cost, GHA features):

```bash
bash scripts/bash/run_supplementary_analyses.sh
```

### Step 1 — Reproduce open-weight experiments (Unsloth)

```bash
export MODELS_PATH=/path/to/UNSLOTH_MODELS

# RQ1: zero-shot & few-shot on Sample_153/Set_Full (Gemma, Llama, Mistral)
bash scripts/bash/run_cigrate_experiments.sh rq1

# RQ2: 10-fold CV fine-tuning on Sample_153/Set_90_10 (Gemma)
bash scripts/bash/run_cigrate_experiments.sh rq2-cv

# RQ2: final models on Sample_153/Set_100_0 (Gemma)
bash scripts/bash/run_cigrate_experiments.sh rq2-final

# RQ4: PR migrations on Sample_30_30/Set_30_30 (requires Set_100_0 weights)
bash scripts/bash/run_cigrate_experiments.sh rq4
```

Or run a single Unsloth job manually (mirrors the SLURM wrapper):

```bash
python scripts/fine_tuning/CIgrate_fine_tune_unsloth_with_final_hyperparameters.py \
  "$MODELS_PATH" "google/gemma-3-12b-it" 4096 True \
  data \
  data/CIgrate_FineTuning_Data/OurCleanData/Sample_153/Set_Full \
  results/recomputed_outputs/CIgrate_Results_ZeroShot/Sample_153/Set_Full \
  travis_to_gha zero_shot
```

Arguments: `models_path model_name max_seq_length load_4bit config_path input_data_path output_data_path migration_type mode`

### Step 2 — Reproduce GPT-4o experiments

```bash
export OPENAI_API_KEY=your_key_here
python scripts/proprietary/CIgrate_gpt-4o.py
```

### Step 3 — Recompute metrics and statistical tests

```bash
# All generated migrations live here:
#   CIgrate_Results_ZeroShot|FewShot|FineTuned  (OurCleanData, 153 projects)
#   CIMig_Results                                (noisy baseline)
RESULTS_BASE=results/generated_outputs/CIgrate_New_Results-OurCleanData-FinalHyperparameters

python scripts/analysis/compute_similarity.py \
  --migrations travis_to_gha gha_to_travis \
  --prompt-mode fine-tuned zero-shot few-shot cimig \
  --results-base "$RESULTS_BASE"

python scripts/analysis/wilcoxon_across_modes_and_models.py \
  --csv results/similarity/similarity_result.csv \
  --mode-a zero-shot --mode-b few-shot
```

### Step 4 — Linting

```bash
python scripts/analysis/lint_ci_configs.py \
  --base results/generated_outputs/CIgrate_New_Results-OurCleanData-FinalHyperparameters
```

See `scripts/analysis/LINTING_README.md` for scoring details.

## Data

### Curated dataset (`OurCleanData`)

Two senior DevOps engineers manually created functionally equivalent GitHub Actions workflows for each Travis CI configuration.

- **153 projects** (`Sample_153`): primary evaluation set
  - `Set_Full/instruction_dataset_{travis_to_gha,gha_to_travis}.csv` — prompting
  - `Set_90_10/Set_90_10_V_{01..10}/` — 10-fold CV train/test splits
  - `Set_100_0/` — train on all 153, for final deployment models
- **30 projects** (`Sample_30_30`): pull-request study
  - `Set_30_30/` — projects for RQ4

### Supplementary materials

- `data/supplementary/Pull_Requests.xlsx` — RQ4 PR metadata
- `data/supplementary/TravisCI_and_GitHubActions_Pairs_Samples.docx` — sample migration pairs

## Results Index

| File | Description |
|------|-------------|
| `results/similarity/similarity_result.csv` | Per-project similarity scores |
| `results/similarity/summary_by_model.csv` | Aggregated means per model/mode/direction |
| `results/similarity/noisy_dataset_comparison.csv` | 251-project CIMig data comparison |
| `results/linting/linting_detailed_results.csv` | Per-file lint scores |
| `results/analysis/llama_failure_mode_summary.csv` | Llama 3.1 8B failure taxonomy |
| `results/analysis/gpt4o_cost_estimate.csv` | GPT-4o token counts and API cost |
| `results/analysis/gha_feature_summary.csv` | GHA workflow features vs cosine |
| `results/statistics/*.csv` | Precomputed significance tests |
| `results/tables/*.tex` | LaTeX tables |
| `results/generated_outputs/` | OurCleanData generated migrations |

## Docker Setup (optional)

```bash
docker build -t cigrate-replication .
docker run --gpus all -it --rm -e OPENAI_API_KEY=... cigrate-replication
```

## Fine-tuned model weights

Fine-tuned model weights are not included due to size. Contact the authors if interested.

## License

Code: **MIT License**. Data: **CC BY 4.0** unless otherwise noted. See `LICENSE`.

## How to Cite

```bibtex
@article{hossain2026cigrate,
  title={CIgrate: Automating CI Service Migration with Large Language Models},
  author={Hossain, Md Nazmul and Ghaleb, Taher A.},
  journal={Submitted to Empirical Software Engineering},
  year={2026},
  publisher={Springer}
}
```

## Contact

For questions about this replication package, open an issue or contact the authors of the paper.
