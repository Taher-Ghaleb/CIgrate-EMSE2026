#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank tests between two prompt modes in similarity_result.csv.

Rows are paired on (project_name, migration_direction, model_name, set_folder_name);
only keys present in BOTH modes are used (inner join).

Strata written to CSV:
  - all_pairs
  - model=<name>          (each model after filters)
  - migration=<direction>
  - model=<name>|migration=<direction>

Usage:
  python wilcoxon_across_modes_and_models.py similarity_result.csv
  python wilcoxon_across_modes_and_models.py --csv /path/to/similarity_result.csv
  python wilcoxon_across_modes_and_models.py -i similarity_result.csv \\
      --mode-a few-shot --mode-b few-shot-dynamic-examples --models gemma3_12b llama3_1_8b
  python wilcoxon_across_modes_and_models.py --csv path/to.csv --migrations travis_to_gha \\
      --mode-a zero-shot --mode-b few-shot --out report.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


KEY_COLS = ["project_name", "migration_direction", "model_name", "set_folder_name"]
METRICS = [
    "max_cosine_similarity",
    "max_crystal_bleu",
    "avg_cosine_similarity",
    "avg_crystal_bleu",
]


def _slug(mode: str) -> str:
    """Suffix for merged columns (alphanumeric + underscore)."""
    s = mode.strip().replace("-", "_")
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    return s or "mode"


def _paired_merge(df: pd.DataFrame, mode_a: str, mode_b: str) -> tuple[pd.DataFrame, str, str]:
    left = df[df["prompt_mode"] == mode_a][KEY_COLS + METRICS].copy()
    right = df[df["prompt_mode"] == mode_b][KEY_COLS + METRICS].copy()
    slug_a = _slug(mode_a)
    slug_b = _slug(mode_b)
    if slug_a == slug_b:
        slug_a, slug_b = f"{slug_a}_left", f"{slug_b}_right"
    merged = left.merge(
        right,
        on=KEY_COLS,
        suffixes=(f"_{slug_a}", f"_{slug_b}"),
        how="inner",
    )
    return merged, slug_a, slug_b


def _wilcoxon_paired(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    """Wilcoxon on (x - y): median of differences tests symmetric about 0."""
    x = x.astype(float)
    y = y.astype(float)
    n_nz = int((x - y != 0).sum())
    if n_nz == 0:
        return None, None, 0
    try:
        res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        stat = float(res.statistic) if res.statistic is not None else None
        p = float(res.pvalue) if res.pvalue is not None else None
        return stat, p, n_nz
    except ValueError:
        return None, None, n_nz


def run_strata(
    merged: pd.DataFrame,
    label: str,
    mask: pd.Series,
    slug_a: str,
    slug_b: str,
    mode_a: str,
    mode_b: str,
) -> list[dict]:
    sub = merged.loc[mask]
    rows_out: list[dict] = []
    if len(sub) < 6:
        for m in METRICS:
            rows_out.append(
                {
                    "stratum": label,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "n_pairs": len(sub),
                    "metric": m,
                    f"median_diff_{slug_b}_minus_{slug_a}": None,
                    "statistic": None,
                    "pvalue": None,
                    "note": "n<6 Wilcoxon unreliable; skipped",
                }
            )
        return rows_out

    med_col = f"median_diff_{slug_b}_minus_{slug_a}"
    for m in METRICS:
        col_b = f"{m}_{slug_b}"
        col_a = f"{m}_{slug_a}"
        xv = sub[col_b]
        yv = sub[col_a]
        med_diff = float((xv - yv).median())
        stat, p, n_nz = _wilcoxon_paired(xv, yv)
        rows_out.append(
            {
                "stratum": label,
                "mode_a": mode_a,
                "mode_b": mode_b,
                "n_pairs": len(sub),
                "metric": m,
                med_col: med_diff,
                "statistic": stat,
                "pvalue": p,
                "n_nonzero_diff": n_nz,
                "note": "",
            }
        )
    return rows_out


def _normalize_report_columns(rows: list[dict], slug_b: str, slug_a: str) -> list[dict]:
    """Use a single canonical column name for median diff in CSV."""
    key = f"median_diff_{slug_b}_minus_{slug_a}"
    out = []
    for r in rows:
        r2 = {k: v for k, v in r.items() if not k.startswith("median_diff_") or k == key}
        if key in r:
            r2["median_diff_mode_b_minus_mode_a"] = r.get(key)
        out.append(r2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired Wilcoxon signed-rank: compare two prompt modes (model- and migration-wise strata)."
    )
    ap.add_argument(
        "--csv",
        "-i",
        dest="csv_opt",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to similarity_result.csv (overrides positional CSV if both given)",
    )
    ap.add_argument(
        "csv_positional",
        type=Path,
        nargs="?",
        default=None,
        metavar="CSV",
        help="Path to similarity_result.csv (default: ./similarity_result.csv if omitted)",
    )
    ap.add_argument(
        "--mode-a",
        type=str,
        default="few-shot",
        metavar="MODE",
        help="First prompt_mode (baseline for difference: mode_b - mode_a)",
    )
    ap.add_argument(
        "--mode-b",
        type=str,
        default="few-shot-dynamic-examples",
        metavar="MODE",
        help="Second prompt_mode (e.g. few-shot-dynamic-examples)",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="If set, only these model_name values (e.g. gemma3_12b mistral_7b). Omit for all models in CSV.",
    )
    ap.add_argument(
        "--migrations",
        nargs="*",
        default=None,
        metavar="DIR",
        help="If set, only these migration_direction values (e.g. travis_to_gha gha_to_travis). Omit for all.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default under csv dir from mode and optional model filter)",
    )
    args = ap.parse_args()
    csv_in = args.csv_opt or args.csv_positional or Path("similarity_result.csv")
    csv_path = csv_in.expanduser().resolve()
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    for c in KEY_COLS + ["prompt_mode"] + METRICS:
        if c not in df.columns:
            print(f"Missing column {c!r} in CSV", file=sys.stderr)
            return 1

    mode_a = args.mode_a.strip()
    mode_b = args.mode_b.strip()
    if mode_a == mode_b:
        print("--mode-a and --mode-b must differ.", file=sys.stderr)
        return 1

    for m in (mode_a, mode_b):
        if m not in df["prompt_mode"].unique():
            print(
                f"Warning: no rows with prompt_mode={m!r} (present: {sorted(df['prompt_mode'].unique())})",
                file=sys.stderr,
            )

    if args.models is not None and len(args.models) > 0:
        allowed = set(args.models)
        df = df[df["model_name"].isin(allowed)].copy()
        if df.empty:
            print(f"No rows after --models filter {args.models!r}", file=sys.stderr)
            return 1

    if args.migrations is not None and len(args.migrations) > 0:
        allowed_m = set(args.migrations)
        df = df[df["migration_direction"].isin(allowed_m)].copy()
        if df.empty:
            print(f"No rows after --migrations filter {args.migrations!r}", file=sys.stderr)
            return 1

    merged, slug_a, slug_b = _paired_merge(df, mode_a, mode_b)
    n = len(merged)
    if n == 0:
        print(
            f"No overlapping rows for prompt_mode {mode_a!r} vs {mode_b!r} after filters.",
            file=sys.stderr,
        )
        return 1

    all_rows: list[dict] = []

    all_rows.extend(
        run_strata(
            merged,
            "all_pairs",
            pd.Series(True, index=merged.index),
            slug_a,
            slug_b,
            mode_a,
            mode_b,
        )
    )

    for mig in sorted(merged["migration_direction"].unique()):
        all_rows.extend(
            run_strata(
                merged,
                f"migration={mig}",
                merged["migration_direction"] == mig,
                slug_a,
                slug_b,
                mode_a,
                mode_b,
            )
        )

    for model in sorted(merged["model_name"].unique()):
        all_rows.extend(
            run_strata(
                merged,
                f"model={model}",
                merged["model_name"] == model,
                slug_a,
                slug_b,
                mode_a,
                mode_b,
            )
        )

    for model in sorted(merged["model_name"].unique()):
        for mig in sorted(merged["migration_direction"].unique()):
            mask = (merged["model_name"] == model) & (merged["migration_direction"] == mig)
            if not mask.any():
                continue
            all_rows.extend(
                run_strata(
                    merged,
                    f"model={model}|migration={mig}",
                    mask,
                    slug_a,
                    slug_b,
                    mode_a,
                    mode_b,
                )
            )

    all_rows = _normalize_report_columns(all_rows, slug_b, slug_a)
    report = pd.DataFrame(all_rows)

    out_path = args.out
    if out_path is None:
        parts = ["wilcoxon", slug_a, "vs", slug_b]
        if args.models:
            parts.append("_".join(sorted(args.models)))
        out_path = csv_path.parent / ("_".join(parts) + ".csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    print(f"Input: {csv_path}")
    print(f"Compare: {mode_b!r} vs {mode_a!r}  (median_diff = mode_b - mode_a)")
    if args.models:
        print(f"Models filter: {args.models}")
    if args.migrations:
        print(f"Migrations filter: {args.migrations}")
    print(f"Paired rows: {n}")
    print(merged["model_name"].value_counts().to_string())
    print("\n=== all_pairs ===")
    sub = report[report["stratum"] == "all_pairs"]
    for _, r in sub.iterrows():
        if r.get("note"):
            print(f"  {r['metric']}: {r['note']}")
        elif r.get("pvalue") is not None:
            print(
                f"  {r['metric']}: n={r['n_pairs']} median_diff={r['median_diff_mode_b_minus_mode_a']:.4f} "
                f"p={r['pvalue']:.4g}"
            )
        else:
            print(f"  {r['metric']}: n={r['n_pairs']}")
    print(f"\nFull table: {out_path}")
    print("Note: p-values are not multiplicity-adjusted across metrics/strata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
