# CI Configuration Linting Tool

Automated linting and validation tool for CI/CD configuration files (GitHub Actions & Travis CI) across multiple LLM-generated outputs.

## Scoring System

Each file receives a composite score (0.0-1.0) based on three components:

- **YAML Parsing (40%)**: Valid YAML syntax via PyYAML
- **Schema Validation (30%)**: Structural requirements (jobs, steps, triggers, etc.)
- **External Linting (30%)**: Service-specific validation
  - GitHub Actions: `actionlint` (lenient mode - fails only on syntax errors)
  - Travis CI: `travis lint` command

**Formula**: `composite_score = yaml_score × 0.4 + schema_score × 0.3 + external_score × 0.3`

## Requirements

```bash
# Python dependencies
pip install pyyaml tqdm

# External tools
brew install actionlint        # For GitHub Actions
gem install travis             # For Travis CI
```

## Usage

```bash
# Basic usage (scans all settings)
python lint_ci_configs.py

# Skip original ground truth files
python lint_ci_configs.py --skip-originals

# Specify custom results directory
python lint_ci_configs.py --base /path/to/results
```

## Output Files

All outputs are timestamped CSVs in `_lint_results/`:

### Primary Files
- `lint_detailed_{timestamp}.csv` - All individual file results with scores
- `scorecard_by_setting_direction_model_{timestamp}.csv` - Model performance by setting/direction
- `summary_by_setting_direction_{timestamp}.csv` - Setting-level comparison (Zero-Shot vs Few-Shot vs Fine-Tuned)

### Additional Aggregations
- `stats_by_setting_direction_model_{timestamp}.csv` - Statistical distribution (min/max/quartiles)
- `ranked_models_by_setting_direction_{timestamp}.csv` - Model rankings per setting/direction
- `best_model_by_setting_direction_{timestamp}.csv` - Best model by median score
- `external_coverage_by_setting_direction_service_{timestamp}.csv` - External tool coverage
- `tools_by_project_direction_{timestamp}.csv` - Per-project tool pass rates
- `agg_by_setting_direction_model_service_{timestamp}.csv` - Model aggregates
- `agg_by_project_and_setting_{timestamp}.csv` - Project-level by setting
- `agg_by_project_setting_direction_model_service_{timestamp}.csv` - Detailed project breakdowns
- `agg_overall_by_project_{timestamp}.csv` - Overall project performance

## Supported Settings

- **Zero-Shot**: LLM without examples
- **Few-Shot**: LLM with example prompts
- **Fine-Tuned**: Fine-tuned LLM models (supports nested model folder structure)
- **CIMig**: Baseline migration tool

## Directory Structure Expected for Results

```
Results/
├── CIgrate_Results_ZeroShot/
│   └── project1/
│       └── 03_CIgrate_Travis_to_GHA/
│           └── model1/
│               └── actions.yml
├── CIgrate_Results_FewShot/
├── CIgrate_Results_FineTuned/
│   └── model_folder_name/          # Model folder at top level
│       └── project1/
│           └── 03_CIgrate_Travis_to_GHA/
│               └── model_subfolder/  # Model name extracted from here
│                   └── actions.yml
└── CIgrate_Results_CIMig/
```

## Key Features

- **Lenient Actionlint**: Passes on version warnings, fails only on syntax errors
- **Fine-Tuned Support**: Handles nested model folder structure
- **Component Visibility**: All aggregations show yaml_score, schema_score, external_score
- **Progress Tracking**: Real-time progress bar showing iterations/rate
