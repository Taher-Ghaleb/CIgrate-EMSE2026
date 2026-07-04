#!/usr/bin/env python3
"""
Automated failure-mode classification for Llama 3.1 8B outputs.
Mirrors the open-coding categories from the paper TODO:
  - markdown_fence_with_preamble
  - markdown_fence_only
  - conversational_preamble_only
  - few_shot_example_echo
  - input_echo_not_migrated
  - truncated_or_too_short
  - duplicate_or_hallucinated_keys
  - other_yaml_structure_error
  - valid_yaml
  - empty_output
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_BASE = (
    PACKAGE_ROOT
    / "results/generated_outputs/extracted/CIgrate_New_Results-OurCleanData-FinalHyperparameters"
)
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results/analysis"

FEW_SHOT_SIGNATURES = [
    "openjdk8",
    "oraclejdk9",
    "codecov.io/bash",
    "distribution: temurin",
    "jdk: ['8', '9']",
]

PREAMBLE_PATTERNS = [
    r"^Here is the",
    r"^This GitHub Actions",
    r"^This Travis",
    r"^The migrated",
    r"^Migrated",
]


def classify_output(text: str, direction: str, input_text: str = "") -> str:
    t = (text or "").strip()
    if not t:
        return "empty_output"

    has_preamble = any(re.match(p, t.lstrip(), re.I) for p in PREAMBLE_PATTERNS)
    has_fence = "```" in t

    stripped = re.sub(r"^```ya?ml?\s*", "", t, flags=re.I)
    stripped = re.sub(r"\s*```\s*$", "", stripped.strip())

    echo_hits = sum(1 for s in FEW_SHOT_SIGNATURES if s in t)
    if echo_hits >= 2:
        return "few_shot_example_echo"

    if input_text:
        in_lines = [ln.strip() for ln in input_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if in_lines:
            matched = sum(1 for ln in in_lines[:12] if ln in t)
            if matched >= min(5, int(len(in_lines[:12]) * 0.7)):
                return "input_echo_not_migrated"

    try:
        yaml.safe_load(stripped if stripped else t)
        return "valid_yaml"
    except Exception:
        pass

    if has_preamble and has_fence:
        return "markdown_fence_with_preamble"
    if has_fence:
        return "markdown_fence_only"
    if has_preamble:
        return "conversational_preamble_only"
    if len(t.splitlines()) < 8:
        return "truncated_or_too_short"

    key_names = re.findall(r"^(\s*)([\w.-]+):", t, re.M)
    names = [k[1] for k in key_names]
    if len(names) != len(set(names)):
        return "duplicate_or_hallucinated_keys"

    return "other_yaml_structure_error"


def analyze_mode(results_base: Path, mode: str) -> list[dict]:
    folder = "CIgrate_Results_FewShot" if mode == "few-shot" else "CIgrate_Results_ZeroShot"
    root = results_base / folder / "Sample_153" / "Set_Full"
    rows = []

    for proj in sorted(root.iterdir()):
        if not proj.is_dir():
            continue
        for direction, sub, out_name, in_name in [
            ("travis_to_gha", "03_CIgrate_Travis_to_GHA", "actions.yml", "01_Original_Travis/travis.yml"),
            ("gha_to_travis", "04_CIgrate_GHA_to_Travis", "travis.yml", "02_Original_GHA/actions.yml"),
        ]:
            out_path = proj / sub / "llama3_1_8b" / out_name
            if not out_path.exists():
                continue
            inp_path = proj / in_name
            text = out_path.read_text(encoding="utf-8", errors="ignore")
            inp = inp_path.read_text(encoding="utf-8", errors="ignore") if inp_path.exists() else ""
            cat = classify_output(text, direction, inp)
            rows.append(
                {
                    "project": proj.name,
                    "direction": direction,
                    "mode": mode,
                    "category": cat,
                    "valid_yaml": cat == "valid_yaml",
                    "input_lines": len(inp.splitlines()),
                    "output_lines": len(text.splitlines()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", type=Path, default=DEFAULT_RESULTS_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    all_rows = analyze_mode(args.results_base, "few-shot") + analyze_mode(args.results_base, "zero-shot")
    out_csv = args.output_dir / "llama_failure_mode_analysis.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    summary_path = args.output_dir / "llama_failure_mode_summary.csv"
    summary_rows = []
    for mode in ["few-shot", "zero-shot"]:
        sub = [r for r in all_rows if r["mode"] == mode]
        for direction in ["travis_to_gha", "gha_to_travis"]:
            dsub = [r for r in sub if r["direction"] == direction]
            fails = [r for r in dsub if not r["valid_yaml"]]
            cats = Counter(r["category"] for r in fails)
            for cat, n in cats.most_common():
                summary_rows.append(
                    {
                        "mode": mode,
                        "direction": direction,
                        "category": cat,
                        "count": n,
                        "pct_of_failures": round(100 * n / max(1, len(fails)), 1),
                        "total_in_group": len(dsub),
                    }
                )

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote {out_csv} ({len(all_rows)} rows)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
