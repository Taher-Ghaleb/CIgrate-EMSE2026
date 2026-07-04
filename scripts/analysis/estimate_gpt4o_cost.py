#!/usr/bin/env python3
"""
Estimate GPT-4o token usage and API cost using tiktoken, mirroring CIgrate_gpt-4o.py prompt construction.

Counts tokens for each project in the instruction CSVs (system + user messages)
and uses saved gpt-4o outputs as completion tokens.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import tiktoken

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/analysis/gpt4o_cost_estimate.csv"

# OpenAI published rates (USD per 1M tokens) — update date if pricing changes
# Source: https://openai.com/api/pricing/ (gpt-4o as of early 2025)
GPT4O_INPUT_PER_M = 2.50
GPT4O_OUTPUT_PER_M = 10.00

ENC = tiktoken.encoding_for_model("gpt-4o")


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_messages(config_dir: Path, migration: str, mode: str, input_yaml: str) -> tuple[str, str]:
    if migration == "travis_to_gha":
        system = read_prompt(config_dir / "prompts/travis_to_gha_system.txt")
        user_tpl = read_prompt(config_dir / "prompts/travis_to_gha_user.txt")
        ex_file = config_dir / "few-shot-examples/travis_to_gha.txt"
    else:
        system = read_prompt(config_dir / "prompts/gha_to_travis_system.txt")
        user_tpl = read_prompt(config_dir / "prompts/gha_to_travis_user.txt")
        ex_file = config_dir / "few-shot-examples/gha_to_travis.txt"

    examples = read_prompt(ex_file) if mode == "few-shot" else ""
    user = user_tpl.replace("{{ FEW_SHOT_EXAMPLES }}", examples).replace("{{ INPUT_CODE }}", input_yaml)
    return system, user


def count_tokens(text: str) -> int:
    return len(ENC.encode(text or ""))


def extract_input_from_row(row: dict) -> str:
    raw = row.get("input", "")
    prefixes = [
        "Migrate this TRAVIS TO GHA configuration:\n\n",
        "Migrate this GHA TO TRAVIS configuration:\n\n",
    ]
    cleaned = raw.strip()
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p) :].strip()
            break
    cleaned = re.sub(r"(secure:\s*)(\S+)", r'\1"********************"', cleaned)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument(
        "--gpt-results",
        type=Path,
        default=PACKAGE_ROOT.parent / "CIgrate_New_Results-OurCleanData-GPT-4o",
        help="GPT-4o outputs (sibling of ReplicationPackage by default; optional for cost recompute)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_dir = args.package_root / "data"
    rows_out = []

    for migration, csv_name in [
        ("travis_to_gha", "instruction_dataset_travis_to_gha.csv"),
        ("gha_to_travis", "instruction_dataset_gha_to_travis.csv"),
    ]:
        csv_path = args.package_root / "data/CIgrate_FineTuning_Data/OurCleanData/Sample_153/Set_Full" / csv_name
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                project = row["project"]
                input_yaml = extract_input_from_row(row)
                for mode in ["zero-shot", "few-shot"]:
                    system, user = build_messages(config_dir, migration, mode.replace("-", "_"), input_yaml)
                    in_tok = count_tokens(system) + count_tokens(user)

                    # completion from saved output if present
                    sub = "03_CIgrate_Travis_to_GHA" if migration == "travis_to_gha" else "04_CIgrate_GHA_to_Travis"
                    out_name = "actions.yml" if migration == "travis_to_gha" else "travis.yml"
                    folder = "CIgrate_Results_FewShot" if mode == "few-shot" else "CIgrate_Results_ZeroShot"
                    out_path = (
                        args.gpt_results
                        / folder
                        / "Sample_153"
                        / "Set_Full"
                        / project
                        / sub
                        / "gpt-4o"
                        / out_name
                    )
                    out_text = out_path.read_text(encoding="utf-8", errors="ignore") if out_path.exists() else row.get("output", "")
                    out_tok = count_tokens(out_text)

                    in_cost = in_tok / 1e6 * GPT4O_INPUT_PER_M
                    out_cost = out_tok / 1e6 * GPT4O_OUTPUT_PER_M
                    rows_out.append(
                        {
                            "project": project,
                            "migration": migration,
                            "mode": mode,
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "total_tokens": in_tok + out_tok,
                            "cost_usd": round(in_cost + out_cost, 5),
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # aggregate
    from collections import defaultdict

    agg = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "n": 0})
    for r in rows_out:
        key = (r["migration"], r["mode"])
        agg[key]["tokens"] += r["total_tokens"]
        agg[key]["cost"] += r["cost_usd"]
        agg[key]["n"] += 1

    print("GPT-4o cost estimate (tiktoken, published list rates):")
    for key, v in sorted(agg.items()):
        print(f"  {key[0]} {key[1]}: {v['n']} projects, {v['tokens']:,} tokens, ${v['cost']:.2f}")
    total_cost = sum(v["cost"] for v in agg.values())
    print(f"  ALL (both dirs, both modes): ${total_cost:.2f}")
    print(f"Wrote per-project breakdown to {args.output}")


if __name__ == "__main__":
    main()
