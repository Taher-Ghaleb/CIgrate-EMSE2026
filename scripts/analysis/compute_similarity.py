"""
Compute Cosine Similarity and CrystalBLEU between original and migrated CI configs.

CLI:
    --migrations: one or more of travis_to_gha, gha_to_travis
    --prompt-mode: one of zero-shot, few-shot, few-shot-dynamic-examples, fine-tuned, cimig
    run command: python -m compute_similarity --migrations travis_to_gha gha_to_travis --prompt-mode fine-tuned zero-shot few-shot few-shot-dynamic-examples cimig
    run command with custom paths: python -u CIgrate_Similarity/compute_similarity.py --migrations travis_to_gha gha_to_travis --prompt-mode fine-tuned zero-shot few-shot few-shot-dynamic-examples cimig --results-base /home/tghaleb/CI_LLM/CIgrate_New_Results-OurCleanData --output-base /home/tghaleb/CI_LLM/All_Similarity_Output

    readable command:
        python -u compute_similarity.py \
            --migrations travis_to_gha gha_to_travis \
            --prompt-mode fine-tuned zero-shot few-shot few-shot-dynamic-examples cimig \
            --results-base /home/tghaleb/CI_LLM/CIgrate_New_Results-OurCleanData \
            --output-base /home/tghaleb/CI_LLM/CIgrate_Similarity/All_Similarity_Results

Outputs:
    1) Code/Similarity/Similarity_results/similarity_result_<ts>.csv (per project/model rows)
    2) Code/Similarity/Similarity_results/summary_by_model_<ts>.csv (averages per mode/model family incl. CIMig)
    3) Code/Similarity/Similarity_results/side_by_side_<mode>_<ts>.csv (wide table per project with models + CIMig side-by-side)
"""

from __future__ import annotations

import argparse
import csv
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from collections import defaultdict
from statistics import mean, stdev, median

from cosine_similarity import textToMap, cosineSimilarity
from crystalbleu_similarity import (
    crystal_bleu,
    build_ignore_ngrams,
)

# ----------------------------- settings -----------------------------

# Default base directories (can be overridden via CLI args)
DEFAULT_RESULTS_BASE = Path("Data_Source")
DEFAULT_OUTPUT_BASE = Path("All_Similarity_Output")

# Source folder mappings
DEFAULT_SOURCE_FOLDERS = {
    "zero-shot": "CIgrate_Results_ZeroShot",
    "few-shot": "CIgrate_Results_FewShot",
    "few-shot-dynamic-examples": "CIgrate_Results_FewShotDynamicExamples",
    "fine-tuned": "CIgrate_Results_FineTuned",
    "cimig": "CIMig_Results",
}


# ----------------------------- utils -----------------------------

KEYWORDS = ("build", "compile", "main", "maven", "ci")

# Canonical model family order for side-by-side outputs
MODEL_FAMILIES = ("gemma", "llama", "mistral", "gpt-4", "cimig")

def load_yaml(path: Path) -> str:
    try:
        yaml_text = path.read_text(encoding="utf-8", errors="ignore")
        yaml_text = yaml_text.replace("\r\n", "\n").replace("\r", "\n")
        yaml_text = re.sub(r'(secure:\s*)(\S+)', r'\1********************', yaml_text)
        return "\n".join(line.strip() for line in yaml_text.split("\n")).strip()
    except FileNotFoundError:
        logging.warning("Missing file: %s", path)
        return ""
    except Exception as e:  # pragma: no cover
        logging.warning("Failed reading %s: %s", path, e)
        return ""


def compute_cosine_similarity(a: str, b: str) -> float:
    score = cosineSimilarity(textToMap(a), textToMap(b))
    # Treat invalid scores (2.0 = empty/zero-norm vectors) as 0.0
    return 0.0 if score == 2.0 else score


def compute_crystal_bleu(a: str, b: str) -> float:
    # Build a small ignore set from both texts (CrystalBLEU behavior)
    ignore = build_ignore_ngrams([a, b], max_n=2, top_k=1)
    return float(crystal_bleu(a, [b], ignore_ngrams=ignore, max_n=2, smooth=True))


def aggregate_scores(scores: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not scores:
        return (0.0, 0.0, 0.0, 0.0)
    cos_vals = [c for c, _ in scores]
    bleu_vals = [b for _, b in scores]
    avg_cos = sum(cos_vals) / len(cos_vals)
    max_cos = max(cos_vals)
    avg_bleu = sum(bleu_vals) / len(bleu_vals)
    max_bleu = max(bleu_vals)
    return (avg_cos, max_cos, avg_bleu, max_bleu)


def canonical_family(model_name: str) -> str:
    """Normalize a model or directory name into a consistent, lowercase format.

    This function standardizes naming by converting special characters 
    (e.g., '/' → '_', others → '-') and ensuring uniform casing.

    Examples:
      - meta/llama-3-70b  -> meta_llama-3-70b
      - google/gemma-2b   -> google_gemma-2b
      - mistral-7b        -> mistral-7b
      - openai/gpt-4o-mini -> openai_gpt-4o-mini
      - CIMig              -> cimig
    """
    n = model_name.strip().lower()
    if (n == "cimig"):
        return "cimig"
    n = model_name.strip().lower()
    n = n.replace('/', '_')
    n = re.sub(r'[^a-z0-9_-]', '-', n)
    return n


# --------------------------- selectors ---------------------------

def select_main_gha_file(files: list[Path]) -> Optional[Path]:
    """
    Mirrors the Java logic exactly:
      1) Scan in order and set main file to the LAST path whose string contains
         any of the keywords (case-sensitive). Do not break early.
      2) If none matched, pick the first file with the strictly largest size,
         comparing with '>' only. If all sizes are 0 (or list is empty),
         return None.

    Notes:
      - Uses Path.stat().st_size which may raise if a file is missing,
        similar to Files.size(...) throwing in Java.
    """
    main_gh_file: Path | None = None

    # Step 1: keyword scan, last match wins
    for f in files:
        s = str(f)
        if ("build" in s) or ("compile" in s) or ("main" in s) or ("maven" in s) or ("ci" in s):
            main_gh_file = f

    # Step 2: fallback to strictly-largest-by-size
    if main_gh_file is None:
        max_size = 0
        for f in files:
            size = f.stat().st_size  # may raise if missing, like Java Files.size(...)
            if size > max_size:
                main_gh_file = f
                max_size = int(size)

    return main_gh_file


def select_migrated_travis_for_selected_gha(migrated_dir: Path, selected_gha: Path) -> Path | None:
    """
    Given selected_gha (e.g., foo.yml), return:
    - migrated_dir / f"{selected_gha.stem}.travis.yml"
    - if not found, fallback to migrated_dir / "travis.yml"
    If neither exists, return None.
    """
    target = migrated_dir / f"{selected_gha.stem}.travis.yml"
    if target.exists():
        return target

    fallback = migrated_dir / "travis.yml"
    if fallback.exists():
        return fallback

    return None


def select_fallback_migrated_travis(files: list[Path]) -> Path:
    """
    Fallback if no exact match:
      1. Prefer filenames with 'build', 'compile', 'main', 'maven', 'ci'
      2. Else, pick largest file by size
    """
    if not files:
        raise FileNotFoundError("No migrated Travis files found")
    lower_names = {f: f.name.lower() for f in files}
    for kw in KEYWORDS:
        candidates = [f for f, n in lower_names.items() if kw in n]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size if p.exists() else 0)
    return max(files, key=lambda p: p.stat().st_size if p.exists() else 0)


# ------------------------- filesystem ---------------------------

def resolve_base_dir(prompt_mode: str, results_base: Path = DEFAULT_RESULTS_BASE) -> Path:
    if prompt_mode not in DEFAULT_SOURCE_FOLDERS:
        raise ValueError(f"Invalid prompt mode: {prompt_mode}")
    return results_base / DEFAULT_SOURCE_FOLDERS[prompt_mode]


def iter_projects(base_dir: Path, prompt_mode: str) -> Iterable[tuple[Path, str, str]]:
    """Yield (project_dir, set_folder_name, fold_name) tuples.
    For fine-tuned: iterate sample folders, then set folders, then fold folders if any, then project folders within.
    For others: also iterate sample folders, then set folders, then project folders within, no fold folders.
    """
    if not base_dir.exists():
        logging.warning("Base dir doesn't exist: %s", base_dir)
        return []
    
    if prompt_mode == "fine-tuned":
        # We have subfolders for different samples (e.g., Sample_153, Sample_100, Sample_55) with different sets (e.g., Set_90_10, Set_80_20, Set_44_11), and some of them have further subfolders for different folds (e.g., Sample_100/Set_90_10/Set_90_10_V_01, Sample_100/Set_90_10/Set_90_10_V_02, Sample_100/Set_90_10/Set_90_10_V_03, etc.).
        for sample_folder in base_dir.iterdir():
            # Perform similarity measurements only for the final statistically significant sample (Sample_153)
            if sample_folder.name != "Sample_153":
                continue
            if sample_folder.is_dir() and not sample_folder.name.startswith('.'):
                for set_folder in sample_folder.iterdir():
                    if set_folder.is_dir() and not set_folder.name.startswith('.'):
                        for fold_folder in set_folder.iterdir():
                            if fold_folder.is_dir() and not fold_folder.name.startswith('.'):
                                for project in fold_folder.iterdir():
                                    if project.is_dir() and not project.name.startswith('.'):
                                        yield (project, set_folder.name, fold_folder.name)
    else:
        for sample_folder in base_dir.iterdir():
            if sample_folder.is_dir() and not sample_folder.name.startswith('.'):
                for set_folder in sample_folder.iterdir():
                    if set_folder.is_dir() and not set_folder.name.startswith('.'):
                        for project in set_folder.iterdir():
                            if project.is_dir() and not project.name.startswith('.') and not project.name.startswith('_'):
                                yield (project, set_folder.name, "")

def list_original_gha_files(project_dir: Path) -> list[Path]:
    d = project_dir / "02_Original_GHA"
    if not d.exists():
        return []
    return [p for p in d.rglob("*") if p.suffix.lower() in (".yml", ".yaml") and p.is_file()]


# ------------------------ comparisons ---------------------------

def compare_travis_to_gha_cigrate(project_dir: Path, model_dir: Path) -> tuple[float, float, float, float] | None:
    migrated = model_dir / "actions.yml"
    if not migrated.exists():
        logging.warning("Missing migrated actions.yml: %s", migrated)
        return None
    originals = list_original_gha_files(project_dir)
    if not originals:
        logging.warning("No original GHA files: %s", project_dir)
        return None
    migrated_yaml = load_yaml(migrated)
    scores: list[tuple[float, float]] = []
    for original in originals:
        original_yaml = load_yaml(original)
        if not migrated_yaml or not original_yaml:
            continue
        cos = compute_cosine_similarity(migrated_yaml, original_yaml)
        bleu = compute_crystal_bleu(migrated_yaml, original_yaml)
        scores.append((cos, bleu))
    return aggregate_scores(scores)


def compare_gha_to_travis_cigrate(project_dir: Path, model_dir: Path) -> tuple[float, float, float, float] | None:
    # Choose one original GHA
    gha_files = list_original_gha_files(project_dir)
    if not gha_files:
        logging.warning("No original GHA files: %s", project_dir)
        return None
    selected_gha = select_main_gha_file(gha_files)
    migrated_dir = project_dir / "04_CIgrate_GHA_to_Travis" / model_dir.name
    if not migrated_dir.exists():
        logging.warning("Missing migrated dir: %s", migrated_dir)
        return None
    migrated = select_migrated_travis_for_selected_gha(migrated_dir, selected_gha)
    if migrated is None:
        candidates = [p for p in migrated_dir.rglob("*.travis.yml")]
        if not candidates:
            logging.warning("No migrated Travis yml files in: %s", migrated_dir)
            return None
        migrated = select_fallback_migrated_travis(candidates)
    # Original Travis
    original_travis = project_dir / "01_Original_Travis" / "travis.yml"
    if not original_travis.exists():
        logging.warning("Missing original Travis: %s", original_travis)
        return None
    migrated_yaml = load_yaml(migrated)
    original_yaml = load_yaml(original_travis)
    if not migrated_yaml or not original_yaml:
        return None
    cos = compute_cosine_similarity(migrated_yaml, original_yaml)
    bleu = compute_crystal_bleu(migrated_yaml, original_yaml)
    return (cos, cos, bleu, bleu)


def compare_travis_to_gha_cimig(project_dir: Path) -> tuple[float, float, float, float] | None:
    # Migrated: 03_CIMig_Travis_to_GHA/{project_no_ext}_original.yml
    proj_name = project_dir.name
    migrated = project_dir / "03_CIMig_Travis_to_GHA" / f"{proj_name}_original.yml"
    if not migrated.exists():
        logging.warning("Missing CIMig migrated GHA: %s", migrated)
        return None
    gha_files = list_original_gha_files(project_dir)
    if not gha_files:
        logging.warning("No original GHA files: %s", project_dir)
        return None
    selected_gha = select_main_gha_file(gha_files)
    migrated_yaml = load_yaml(migrated)
    selected_gha_yaml = load_yaml(selected_gha)
    if not migrated_yaml or not selected_gha_yaml:
        return None
    cos = compute_cosine_similarity(migrated_yaml, selected_gha_yaml)
    bleu = compute_crystal_bleu(migrated_yaml, selected_gha_yaml)
    return (cos, cos, bleu, bleu)


def compare_gha_to_travis_cimig(project_dir: Path) -> tuple[float, float, float, float] | None:
    # Migrated: 04_CIMig_GHA_to_Travis/{project_no_ext}_travis.yml
    proj_name = project_dir.name
    migrated = project_dir / "04_CIMig_GHA_to_Travis" / f"{proj_name}_travis.yml"
    if not migrated.exists():
        logging.warning("Missing CIMig migrated Travis: %s", migrated)
        return None
    original_travis = project_dir / "01_Original_Travis" / "travis.yml"
    if not original_travis.exists():
        logging.warning("Missing original Travis: %s", original_travis)
        return None
    migrated_yaml = load_yaml(migrated)
    original_yaml = load_yaml(original_travis)
    if not migrated_yaml or not original_yaml:
        return None
    cos = compute_cosine_similarity(migrated_yaml, original_yaml)
    bleu = compute_crystal_bleu(migrated_yaml, original_yaml)
    return (cos, cos, bleu, bleu)


# --------------------------- orchestration ---------------------------

@dataclass
class Row:
    sl: int
    project_name: str
    migration_direction: str
    prompt_mode: str
    model_name: str
    set_folder_name: str 
    fold_name: str
    avg_cosine_similarity: float
    max_cosine_similarity: float
    avg_crystal_bleu: float
    max_crystal_bleu: float


def write_csv(rows: list[Row], outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "SL",
            "project_name",
            "migration_direction",
            "prompt_mode",
            "set_folder_name",
            "fold_name",
            "model_name",
            "avg_cosine_similarity",
            "max_cosine_similarity",
            "avg_crystal_bleu",
            "max_crystal_bleu",
        ])
        for r in rows:
            w.writerow([
                r.sl,
                r.project_name,
                r.migration_direction,
                r.prompt_mode,
                r.set_folder_name,
                r.fold_name,
                r.model_name,
                f"{r.avg_cosine_similarity:.2f}",
                f"{r.max_cosine_similarity:.2f}",
                f"{r.avg_crystal_bleu:.2f}",
                f"{r.max_crystal_bleu:.2f}",
            ])


def parse_args():
    p = argparse.ArgumentParser(description="Compute Cosine and CrystalBLEU similarities")
    p.add_argument(
        "--migrations",
        nargs="+",
        choices=["travis_to_gha", "gha_to_travis"],
        required=True,
        help="Migration directions to process",
    )
    # Accept multiple prompt modes to support running everything in one go
    p.add_argument(
        "--prompt-mode",
        nargs="+",
        choices=["zero-shot", "few-shot", "few-shot-dynamic-examples", "fine-tuned", "cimig"],
        required=True,
        help="Prompt mode(s)",
    )
    p.add_argument(
        "--results-base",
        type=Path,
        default=DEFAULT_RESULTS_BASE,
        help=f"Base directory for result inputs (default: {DEFAULT_RESULTS_BASE})",
    )
    p.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Base directory for output files (default: {DEFAULT_OUTPUT_BASE})",
    )
    return p.parse_args()


def _aggregate_family_per_project(rows: list[Row], include_modes: set[str] | None = None) -> dict[tuple[str, str, str, str, str, str], tuple[float, float]]:
    """Aggregate rows at (project, migration, prompt_mode, set_folder_name, fold_name, family) level, taking the max of
    max_cosine and max_bleu per family within that group (in case multiple variants exist).

    Returns a dict mapping (project_name, migration, prompt_mode, set_folder_name, fold_name, family) -> (max_cos, max_bleu)
    """
    agg: dict[tuple[str, str, str, str, str, str], tuple[float, float]] = {}
    for r in rows:
        if include_modes and r.prompt_mode not in include_modes:
            continue
        fam = canonical_family(r.model_name)
        key = (r.project_name, r.migration_direction, r.prompt_mode, r.set_folder_name, r.fold_name, fam)
        cur = agg.get(key)
        cand = (r.max_cosine_similarity, r.max_crystal_bleu)
        if cur is None or (cand[0] > cur[0] or cand[1] > cur[1]):
            agg[key] = cand
    return agg


def _build_cimig_index(rows: list[Row]) -> dict[tuple[str, str], tuple[float, float]]:
    """Index CIMig results by (project, migration) -> (max_cos, max_bleu)."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for r in rows:
        if canonical_family(r.model_name) == "cimig":
            out[(r.project_name, r.migration_direction)] = (
                r.max_cosine_similarity,
                r.max_crystal_bleu,
            )
    return out


def write_side_by_side_per_mode(rows: list[Row], mode: str, out_dir: Path, ts: str) -> None:
    """Create a wide CSV for the given prompt mode combining models and CIMig side-by-side per project/migration.

    Columns: project_name, migration, set_folder_name, fold_name, model_name, then for each family in MODEL_FAMILIES (excluding CIMig for non-cimig modes),
    two columns: {family}_max_cos, {family}_max_bleu, plus cimig_max_cos/bleu if available.
    Also includes best_model_by_bleu/cos and their scores.
    """
    fam_agg = _aggregate_family_per_project(rows, include_modes={mode})
    cimig_idx = _build_cimig_index(rows)

    # Collect all (project, migration, set_folder_name, fold_name, family) tuples for this mode
    tuples: set[tuple[str, str, str, str, str]] = set((p, m, mf, f, fam) for (p, m, pm, mf, f, fam) in fam_agg.keys() if pm == mode)
    if not tuples:
        return

    families = [f for f in MODEL_FAMILIES if f != "cimig"]
    outfile = out_dir / f"side_by_side_{mode}.csv"
    with outfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["project_name", "migration_direction", "set_folder_name", "fold_name", "model_name", "prompt_mode"]
        for fam in families:
            header += [f"{fam}_max_cos", f"{fam}_max_bleu"]
        header += ["cimig_max_cos", "cimig_max_bleu", "best_model_by_bleu", "best_bleu", "best_model_by_cos", "best_cos"]
        w.writerow(header)

        for project, migration, set_folder_name, fold_name, model_name in sorted(tuples):
            row_vals: list[str | float] = [project, migration, set_folder_name, fold_name, model_name, mode]
            perf: dict[str, tuple[float, float] | None] = {}
            for fam in families:
                key = (project, migration, mode, set_folder_name, fold_name, fam)
                perf[fam] = fam_agg.get(key)
                if perf[fam] is None:
                    row_vals += ["", ""]
                else:
                    row_vals += [f"{perf[fam][0]:.2f}", f"{perf[fam][1]:.2f}"]
            ci = cimig_idx.get((project, migration))
            if ci is None:
                row_vals += ["", ""]
            else:
                row_vals += [f"{ci[0]:.2f}", f"{ci[1]:.2f}"]

            # Determine bests among families + CIMig
            all_perf: list[tuple[str, float, float]] = []
            for fam in families:
                if perf.get(fam) is not None:
                    all_perf.append((fam, perf[fam][0], perf[fam][1]))
            if ci is not None:
                all_perf.append(("cimig", ci[0], ci[1]))

            if all_perf:
                best_bleu = max(all_perf, key=lambda t: t[2])
                best_cos = max(all_perf, key=lambda t: t[1])
                row_vals += [best_bleu[0], f"{best_bleu[2]:.2f}", best_cos[0], f"{best_cos[1]:.2f}"]
            else:
                row_vals += ["", "", "", ""]

            w.writerow(row_vals)


def write_summary_by_model(rows: list[Row], out_dir: Path, ts: str) -> None:
    """Write a summary CSV averaging metrics per (prompt_mode, migration, set_folder_name, fold_name, model_family).

    For the family level, if multiple variants of the same family exist for a project, the best (max) per project is used
    before averaging across projects.
    """
    fam_agg = _aggregate_family_per_project(rows)
    # Collect keys
    by_key: dict[tuple[str, str, str, str, str], list[tuple[float, float]]] = {}
    for (project, migration, pmode, set_folder_name, fold_name, fam), (mc, mb) in fam_agg.items():
        k = (pmode, migration, set_folder_name, fold_name, fam)
        by_key.setdefault(k, []).append((mc, mb))

    outfile = out_dir / f"summary_by_model.csv"
    with outfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "prompt_mode",
            "migration_direction",
            "set_folder_name",
            "fold_name",
            "model_family",
            "projects",
            "avg_of_max_cosine",
            "avg_of_max_crystal_bleu",
        ])
        for (pmode, migration, set_folder_name, fold_name, fam), vals in sorted(by_key.items()):
            if not vals:
                continue
            avg_max_cos = sum(v[0] for v in vals) / len(vals)
            avg_max_bleu = sum(v[1] for v in vals) / len(vals)
            w.writerow([
                pmode,
                migration,
                set_folder_name,
                fold_name,
                fam,
                len(vals),
                f"{avg_max_cos:.2f}",
                f"{avg_max_bleu:.2f}",
            ])


def list_model_dirs(project_dir: Path, prompt_mode: str, migration: str, set_folder_name: str) -> list[Path]:
    if prompt_mode == "cimig":
        return []
    if migration == "travis_to_gha":
        base = project_dir / "03_CIgrate_Travis_to_GHA"
    else:
        base = project_dir / "04_CIgrate_GHA_to_Travis"
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]

def calculate_stats(values):
    """Calculate statistics for a list of values."""
    if not values:
        return {
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
        }
    
    values_float = [float(v) for v in values]
    return {
        'mean': mean(values_float),
        'std': stdev(values_float) if len(values_float) > 1 else 0.0,
        'min': min(values_float),
        'max': max(values_float),
        'median': median(values_float),
    }

def summarize_folds(input_csv: Path, output_csv: Path) -> None:
    """
    Compute statistics across folds for each configuration from the input CSV,
    but only for prompt_mode == 'fine-tuned'. For other modes, copy the input as output as-is.
    """
    # Read the input CSV
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows from {input_csv}")
    
    # Determine if there is a prompt_mode column, and if any rows are for 'fine-tuned'
    has_prompt_mode = rows and 'prompt_mode' in rows[0]
    process_folds = False
    if has_prompt_mode:
        # If any row is fine-tuned, we will summarize by configuration and folds
        process_folds = any(row['prompt_mode'] == 'fine-tuned' for row in rows)
    
    if has_prompt_mode and process_folds:
        # Only include fine-tuned rows
        rows = [row for row in rows if row['prompt_mode'] == 'fine-tuned']

        # Group by configuration: prompt_mode + migration_direction + set_folder_name + model_family
        group_cols = ['prompt_mode', 'migration_direction', 'set_folder_name', 'model_family']
        grouped = defaultdict(lambda: {'cosine': [], 'bleu': []})
        
        for row in rows:
            key = tuple(row[col] for col in group_cols)
            if 'avg_of_max_cosine' in row and row['avg_of_max_cosine']:
                grouped[key]['cosine'].append(row['avg_of_max_cosine'])
            if 'avg_of_max_crystal_bleu' in row and row['avg_of_max_crystal_bleu']:
                grouped[key]['bleu'].append(row['avg_of_max_crystal_bleu'])
        
        summary_stats = []
        for key, values in sorted(grouped.items()):
            config_dict = {col: key[i] for i, col in enumerate(group_cols)}
            num_folds = len(values['cosine']) if values['cosine'] else len(values['bleu'])
            config_dict['num_folds'] = num_folds

            cosine_stats = calculate_stats(values['cosine'])
            config_dict['cosine_mean'] = cosine_stats['mean'] if cosine_stats['mean'] is not None else 0.0
            config_dict['cosine_std'] = cosine_stats['std'] if cosine_stats['std'] is not None else 0.0
            config_dict['cosine_min'] = cosine_stats['min'] if cosine_stats['min'] is not None else 0.0
            config_dict['cosine_max'] = cosine_stats['max'] if cosine_stats['max'] is not None else 0.0
            config_dict['cosine_median'] = cosine_stats['median'] if cosine_stats['median'] is not None else 0.0

            bleu_stats = calculate_stats(values['bleu'])
            config_dict['bleu_mean'] = bleu_stats['mean'] if bleu_stats['mean'] is not None else 0.0
            config_dict['bleu_std'] = bleu_stats['std'] if bleu_stats['std'] is not None else 0.0
            config_dict['bleu_min'] = bleu_stats['min'] if bleu_stats['min'] is not None else 0.0
            config_dict['bleu_max'] = bleu_stats['max'] if bleu_stats['max'] is not None else 0.0
            config_dict['bleu_median'] = bleu_stats['median'] if bleu_stats['median'] is not None else 0.0

            summary_stats.append(config_dict)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        base_cols = group_cols
        metric_cols = [
            'num_folds',
            'cosine_mean', 'cosine_median', 'cosine_std', 'cosine_min', 'cosine_max',
            'bleu_mean', 'bleu_median', 'bleu_std', 'bleu_min', 'bleu_max',
        ]
        fieldnames = base_cols + metric_cols

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stat in summary_stats:
                row = {}
                for col in fieldnames:
                    value = stat.get(col, '')
                    if isinstance(value, float):
                        row[col] = f"{value:.2f}"
                    else:
                        row[col] = value
                writer.writerow(row)
        
        print(f"Summary statistics saved to {output_csv}")
        print(f"Total configurations: {len(summary_stats)}")

    else:
        # For all other prompt_modes, just copy the input to the output
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
            writer = None
            for i, row in enumerate(rows):
                if writer is None:
                    fieldnames = row.keys()
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
        print(f"(No fold summary needed; copied input to {output_csv})")
        print(f"Total configurations: {len(rows)}")
    
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    rows: list[Row] = []
    sl = 1
    for prompt_mode in args.prompt_mode:
        base_dir = resolve_base_dir(prompt_mode, args.results_base)
        for project, set_folder_name, fold_name in iter_projects(base_dir, prompt_mode):
            project_name = project.name
            for migration in args.migrations:
                if prompt_mode == "cimig":
                    if migration == "travis_to_gha":
                        agg = compare_travis_to_gha_cimig(project)
                    else:
                        agg = compare_gha_to_travis_cimig(project)
                    if agg is None:
                        continue
                    avg_cos, max_cos, avg_bleu, max_bleu = agg
                    rows.append(Row(sl, project_name, migration, prompt_mode, "CIMig", "", "", avg_cos, max_cos, avg_bleu, max_bleu))
                    sl += 1
                else:
                    model_dirs = list_model_dirs(project, prompt_mode, migration, set_folder_name)
                    if not model_dirs:
                        logging.warning("No model directories for %s in %s", migration, project)
                        continue
                    for model_dir in model_dirs:
                        model_name = model_dir.name
                        if migration == "travis_to_gha":
                            agg = compare_travis_to_gha_cigrate(project, model_dir)
                        else:
                            agg = compare_gha_to_travis_cigrate(project, model_dir)
                        if agg is None:
                            continue
                        avg_cos, max_cos, avg_bleu, max_bleu = agg
                        rows.append(Row(sl, project_name, migration, prompt_mode, model_name, set_folder_name, fold_name, avg_cos, max_cos, avg_bleu, max_bleu))
                        sl += 1

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out = args.output_base / f"similarity_result.csv"
    write_csv(rows, out)
    logging.info("Wrote %d rows to %s", len(rows), out)

    # Extra comparative outputs
    out_dir = out.parent
    try:
        write_summary_by_model(rows, out_dir, ts)
        logging.info("Wrote summary_by_model for %d rows", len(rows))
        # Create per-mode side-by-side where applicable (exclude 'cimig' itself)
        modes_present = {r.prompt_mode for r in rows if r.prompt_mode != "cimig"}
        for mode in sorted(modes_present):
            write_side_by_side_per_mode(rows, mode, out_dir, ts)
            logging.info("Wrote side_by_side for mode=%s", mode)
    except Exception as e:  # pragma: no cover
        logging.warning("Failed generating summaries: %s", e)

    summarize_folds(out_dir / "summary_by_model.csv", out_dir / "summary_across_folds.csv")
    logging.info("Summarized folds")

