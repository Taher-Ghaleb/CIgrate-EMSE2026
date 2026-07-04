#!/usr/bin/env python3
"""
Extract GHA workflow complexity features for fine-tuning outlier analysis.
Compares bottom vs top cosine-scoring projects for GHA->Travis direction.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMILARITY = PACKAGE_ROOT / "results/similarity/similarity_result.csv"
DEFAULT_RESULTS_BASE = (
    PACKAGE_ROOT
    / "results/generated_outputs/extracted/CIgrate_New_Results-OurCleanData-FinalHyperparameters"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/analysis/gha_feature_analysis.csv"


def extract_features(yaml_text: str) -> dict:
    t = yaml_text or ""
    lines = t.splitlines()
    return {
        "lines": len(lines),
        "jobs": len(re.findall(r"^\s{2}[\w.-]+:\s*$", t, re.M)),
        "steps": t.count("- uses:") + t.count("- run:"),
        "matrix": int("matrix:" in t),
        "services": int("services:" in t),
        "container": int("container:" in t),
        "workflow_call": int("workflow_call" in t),
        "reusable_local_action": int(re.search(r"uses:\s*\./", t) is not None),
        "permissions": int("permissions:" in t),
        "secrets_refs": len(re.findall(r"\$\{\{\s*secrets\.", t)),
        "env_refs": len(re.findall(r"\$\{\{\s*env\.", t)),
        "github_token_refs": int("GITHUB_TOKEN" in t or "github.token" in t),
        "concurrency": int("concurrency:" in t),
        "needs": int("needs:" in t),
        "if_conditions": len(re.findall(r"\n\s+if:\s*", t)),
        "caches": t.count("actions/cache"),
        "deployments": int("deploy:" in t or "environment:" in t),
        "cron_schedule": int("schedule:" in t and "cron:" in t),
        "workflow_dispatch": int("workflow_dispatch" in t),
        "top_level_env": int(re.search(r"^env:", t, re.M) is not None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity-csv", type=Path, default=DEFAULT_SIMILARITY)
    parser.add_argument("--results-base", type=Path, default=DEFAULT_RESULTS_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sim_rows = []
    with args.similarity_csv.open() as f:
        for r in csv.DictReader(f):
            if (
                r["prompt_mode"] == "fine-tuned"
                and r["migration_direction"] == "gha_to_travis"
                and r["model_name"] == "gemma3_12b"
            ):
                sim_rows.append(r)

    enriched = []
    for r in sim_rows:
        gha = (
            args.results_base
            / "CIgrate_Results_FineTuned"
            / "Sample_153"
            / "Set_90_10"
            / r["fold_name"]
            / r["project_name"]
            / "02_Original_GHA"
            / "actions.yml"
        )
        if not gha.exists():
            continue
        feats = extract_features(gha.read_text(encoding="utf-8", errors="ignore"))
        feats.update(
            {
                "project": r["project_name"],
                "fold": r["fold_name"],
                "cosine": float(r["max_cosine_similarity"]),
                "bleu": float(r["max_crystal_bleu"]),
            }
        )
        enriched.append(feats)

    enriched.sort(key=lambda x: x["cosine"])
    k = max(1, len(enriched) // 5)
    low, high = enriched[:k], enriched[-k:]

    feature_names = [k for k in enriched[0].keys() if k not in ("project", "fold", "cosine", "bleu")]

    summary_rows = []
    for feat in feature_names:
        lm = statistics.mean(x[feat] for x in low)
        hm = statistics.mean(x[feat] for x in high)
        summary_rows.append(
            {
                "feature": feat,
                "low_cosine_mean": round(lm, 3),
                "high_cosine_mean": round(hm, 3),
                "delta": round(lm - hm, 3),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        w.writeheader()
        w.writerows(enriched)

    summary_path = args.output.parent / "gha_feature_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(summary_rows, key=lambda r: -abs(r["delta"])))

    print(f"Wrote {args.output} ({len(enriched)} projects)")
    print(f"Wrote {summary_path}")
    print("Top feature deltas (low vs high cosine):")
    for row in sorted(summary_rows, key=lambda r: -abs(r["delta"]))[:8]:
        print(f"  {row['feature']}: low={row['low_cosine_mean']}, high={row['high_cosine_mean']}")


if __name__ == "__main__":
    main()
