#!/usr/bin/env python3
"""
Compute YAML validity and CI linter pass rates for generated migration outputs.

Walks the extracted results tree (see results/generated_outputs/) and reports
per-configuration pass rates matching Table tab:linting-results in the paper.

External tools (optional but recommended):
  - actionlint  — https://github.com/rhysd/actionlint
  - travis-lint — gem install travis

Usage (from ReplicationPackage root):
    python scripts/analysis/compute_linting_metrics.py
    python scripts/analysis/compute_linting_metrics.py --results-dir results/generated_outputs/extracted/CIgrate_New_Results-OurCleanData-FinalHyperparameters
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    PACKAGE_ROOT
    / "results"
    / "generated_outputs"
    / "extracted"
    / "CIgrate_New_Results-OurCleanData-FinalHyperparameters"
)
DEFAULT_OUT = PACKAGE_ROOT / "results" / "linting" / "linting_summary_recomputed.csv"

PROMPT_FOLDERS = {
    "zero-shot": "CIgrate_Results_ZeroShot",
    "few-shot": "CIgrate_Results_FewShot",
    "fine-tuned": "CIgrate_Results_FineTuned",
    "cimig": "CIMig_Results",
}

MODELS = ["gpt-4o", "gemma3_12b", "mistral_7b", "llama3_1_8b"]


@dataclass
class LintStats:
    total: int = 0
    yaml_ok: int = 0
    linter_ok: int = 0

    def add(self, yaml_ok: bool, linter_ok: bool) -> None:
        self.total += 1
        if yaml_ok:
            self.yaml_ok += 1
        if linter_ok:
            self.linter_ok += 1

    @property
    def yaml_pct(self) -> float:
        return 100.0 * self.yaml_ok / self.total if self.total else 0.0

    @property
    def linter_pct(self) -> float:
        return 100.0 * self.linter_ok / self.total if self.total else 0.0


def has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def check_yaml(text: str) -> bool:
    try:
        yaml.safe_load(text)
        return True
    except Exception:
        return False


def run_actionlint(path: Path) -> bool:
    if not has_tool("actionlint"):
        return check_yaml(path.read_text(encoding="utf-8", errors="ignore"))
    proc = subprocess.run(
        ["actionlint", str(path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def run_travis_lint(path: Path) -> bool:
    if not has_tool("travis-lint"):
        return check_yaml(path.read_text(encoding="utf-8", errors="ignore"))
    proc = subprocess.run(
        ["travis-lint", str(path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def lint_file(path: Path, direction: str) -> tuple[bool, bool]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    yaml_ok = check_yaml(text)
    if not yaml_ok:
        return False, False
    if direction == "travis_to_gha":
        return yaml_ok, run_actionlint(path)
    return yaml_ok, run_travis_lint(path)


def iter_outputs(results_dir: Path, prompt_mode: str, model: str | None) -> list[tuple[str, Path]]:
    """Yield (direction, yaml_path) for each generated file."""
    folder = PROMPT_FOLDERS[prompt_mode]
    base = results_dir / folder
    if prompt_mode == "cimig":
        sample = "Sample_251" if (base / "Sample_251").exists() else "Sample_153"
        root = base / sample / "Set_Full"
        paths = []
        for project in sorted(root.iterdir()):
            if not project.is_dir():
                continue
            for direction, sub in [
                ("travis_to_gha", project / "03_CIMig_Travis_to_GHA"),
                ("gha_to_travis", project / "04_CIMig_GHA_to_Travis"),
            ]:
                if sub.exists():
                    for yml in sub.glob("*.yml"):
                        paths.append((direction, yml))
                    for yml in sub.glob("*.yaml"):
                        paths.append((direction, yml))
        return paths

    if prompt_mode == "fine-tuned":
        root = base / "Sample_153" / "Set_90_10"
        paths = []
        for fold in sorted(root.glob("Set_90_10_V_*")):
            for project in sorted(fold.iterdir()):
                if not project.is_dir():
                    continue
                for direction, sub, fname in [
                    ("travis_to_gha", project / "03_CIgrate_Travis_to_GHA" / "gemma3_12b", "actions.yml"),
                    ("gha_to_travis", project / "04_CIgrate_GHA_to_Travis" / "gemma3_12b", "travis.yml"),
                ]:
                    fpath = sub / fname
                    if fpath.exists():
                        paths.append((direction, fpath))
        return paths

    root = base / "Sample_153" / "Set_Full"
    paths = []
    for project in sorted(root.iterdir()):
        if not project.is_dir():
            continue
        for direction, sub_dir, fname in [
            ("travis_to_gha", project / "03_CIgrate_Travis_to_GHA", "actions.yml"),
            ("gha_to_travis", project / "04_CIgrate_GHA_to_Travis", "travis.yml"),
        ]:
            if not sub_dir.exists():
                continue
            fpath = sub_dir / model / fname
            if not fpath.exists():
                for candidate_dir in sub_dir.iterdir():
                    if candidate_dir.is_dir() and (candidate_dir / fname).exists():
                        fpath = candidate_dir / fname
                        break
            if fpath.exists():
                paths.append((direction, fpath))
    return paths


def aggregate(results_dir: Path) -> pd.DataFrame:
    rows = []

    # CIMig baseline (153-project subset when using OurCleanData zip)
    for direction in ["travis_to_gha", "gha_to_travis"]:
        stats = LintStats()
        for d, path in iter_outputs(results_dir, "cimig", None):
            if d != direction:
                continue
            y, l = lint_file(path, d)
            stats.add(y, l)
        rows.append(
            {
                "approach": "CIMig",
                "setting": "rule-based",
                "migration_direction": direction,
                "label": "CIMig",
                "yaml_valid_pct": round(stats.yaml_pct, 1),
                "linter_pass_pct": round(stats.linter_pct, 1),
                "n_files": stats.total,
            }
        )

    for mode in ["zero-shot", "few-shot"]:
        for model in MODELS:
            for direction in ["travis_to_gha", "gha_to_travis"]:
                stats = LintStats()
                for d, path in iter_outputs(results_dir, mode, model):
                    if d != direction:
                        continue
                    y, l = lint_file(path, d)
                    stats.add(y, l)
                if stats.total == 0:
                    continue
                label = f"{model} ({mode})"
                rows.append(
                    {
                        "approach": model,
                        "setting": mode,
                        "migration_direction": direction,
                        "label": label,
                        "yaml_valid_pct": round(stats.yaml_pct, 1),
                        "linter_pass_pct": round(stats.linter_pct, 1),
                        "n_files": stats.total,
                    }
                )

    for direction in ["travis_to_gha", "gha_to_travis"]:
        stats = LintStats()
        for d, path in iter_outputs(results_dir, "fine-tuned", "gemma3_12b"):
            if d != direction:
                continue
            y, l = lint_file(path, d)
            stats.add(y, l)
        rows.append(
            {
                "approach": "gemma3_12b",
                "setting": "fine-tuned",
                "migration_direction": direction,
                "label": "Gemma 3 12B (fine-tuned)",
                "yaml_valid_pct": round(stats.yaml_pct, 1),
                "linter_pass_pct": round(stats.linter_pct, 1),
                "n_files": stats.total,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute linting pass rates for CIgrate outputs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(
            f"Results directory not found: {args.results_dir}\n"
            "Extract results/generated_outputs/CIgrate_New_Results-OurCleanData-FinalHyperparameters.zip first."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = aggregate(args.results_dir)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(df)} rows)")
    if not has_tool("actionlint"):
        print("Note: actionlint not found; GHA linter pass rates fall back to PyYAML-only checks.")
    if not has_tool("travis-lint"):
        print("Note: travis-lint not found; Travis linter pass rates fall back to PyYAML-only checks.")


if __name__ == "__main__":
    main()
