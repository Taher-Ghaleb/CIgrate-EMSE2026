#!/usr/bin/env python3
"""
Regenerate paper figures from precomputed similarity CSVs.

Reads results/similarity/summary_by_model.csv and similarity_result.csv and writes
PDF figures matching those referenced in finalized_EMSE_paper.tex (RQ1–RQ3).

Usage (from ReplicationPackage root):
    python scripts/analysis/generate_paper_figures.py
    python scripts/analysis/generate_paper_figures.py --output-dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_DIR = PACKAGE_ROOT / "results" / "similarity"
DEFAULT_OUT_DIR = PACKAGE_ROOT / "results" / "figures"

MODEL_ORDER = ["gpt-4o", "gemma3_12b", "mistral_7b", "llama3_1_8b"]
MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "gemma3_12b": "Gemma 3 12B",
    "mistral_7b": "Mistral 7B",
    "llama3_1_8b": "Llama 3.1 8B",
}
DIRECTION_LABELS = {
    "travis_to_gha": "Travis → GHA",
    "gha_to_travis": "GHA → Travis",
}


def _round2(x: float) -> float:
    return round(float(x), 2)


def load_tables(sim_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(sim_dir / "summary_by_model.csv")
    detail = pd.read_csv(sim_dir / "similarity_result.csv")
    return summary, detail


def plot_rq1_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    """fig_similarity_comparison.pdf — zero-shot vs few-shot for RQ1."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [("avg_of_max_cosine", "Cosine Similarity"), ("avg_of_max_crystal_bleu", "CrystalBLEU")]
    modes = ["zero-shot", "few-shot"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    for row_idx, direction in enumerate(["travis_to_gha", "gha_to_travis"]):
        for col_idx, (metric_col, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for i, mode in enumerate(modes):
                vals = []
                for model in MODEL_ORDER:
                    row = summary[
                        (summary.prompt_mode == mode)
                        & (summary.migration_direction == direction)
                        & (summary.model_family == model)
                        & (summary.set_folder_name == "Set_Full")
                    ]
                    vals.append(float(row[metric_col].iloc[0]) if len(row) else 0.0)
                ax.bar(x + (i - 0.5) * width, vals, width, label=mode.replace("-", " ").title())

            ax.set_title(f"{DIRECTION_LABELS[direction]} — {metric_name}")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)
            if row_idx == 0 and col_idx == 1:
                ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "fig_similarity_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_finetuning_folds(detail: pd.DataFrame, out_dir: Path) -> None:
    """fig_finetuning_folds_boxplot.pdf — fold-level fine-tuning distributions."""
    ft = detail[
        (detail.prompt_mode == "fine-tuned")
        & (detail.model_name == "gemma3_12b")
        & (detail.set_folder_name == "Set_90_10")
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric_col, title) in zip(
        axes,
        [
            ("max_cosine_similarity", "Cosine Similarity"),
            ("max_crystal_bleu", "CrystalBLEU"),
        ],
    ):
        data = []
        labels = []
        for direction in ["travis_to_gha", "gha_to_travis"]:
            sub = ft[ft.migration_direction == direction][metric_col].values
            data.append(sub)
            labels.append(DIRECTION_LABELS[direction])

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#cfe8ff")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fine-tuned Gemma 3 12B — 10-fold cross-validation", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_finetuning_folds_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)


def _config_means(summary: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (approach label, direction) for RQ3 bar/box plots."""
    rows = []

    cimig = summary[summary.prompt_mode == "cimig"]
    for _, r in cimig.iterrows():
        rows.append(
            {
                "label": "CIMig",
                "direction": r.migration_direction,
                "cosine": r.avg_of_max_cosine,
                "bleu": r.avg_of_max_crystal_bleu,
            }
        )

    for mode in ["zero-shot", "few-shot"]:
        for model in MODEL_ORDER:
            sub = summary[
                (summary.prompt_mode == mode)
                & (summary.model_family == model)
                & (summary.set_folder_name == "Set_Full")
            ]
            for _, r in sub.iterrows():
                label = f"{MODEL_LABELS[model]}\n({mode})"
                rows.append(
                    {
                        "label": label,
                        "direction": r.migration_direction,
                        "cosine": r.avg_of_max_cosine,
                        "bleu": r.avg_of_max_crystal_bleu,
                    }
                )

    ft = summary[
        (summary.prompt_mode == "fine-tuned")
        & (summary.model_family == "gemma3_12b")
        & (summary.set_folder_name == "Set_90_10")
    ]
    for direction in ["travis_to_gha", "gha_to_travis"]:
        sub = ft[ft.migration_direction == direction]
        rows.append(
            {
                "label": "Gemma 3 12B\n(fine-tuned)",
                "direction": direction,
                "cosine": sub.cosine_mean.mean(),
                "bleu": sub.bleu_mean.mean(),
            }
        )

    return pd.DataFrame(rows)


def plot_rq3_bars(config: pd.DataFrame, out_dir: Path) -> None:
    """fig_similarity_comparison-rq1.pdf — all configurations vs CIMig."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row_idx, direction in enumerate(["travis_to_gha", "gha_to_travis"]):
        sub = config[config.direction == direction].reset_index(drop=True)
        labels = sub["label"].tolist()
        x = np.arange(len(labels))
        for col_idx, metric in enumerate(["cosine", "bleu"]):
            ax = axes[row_idx, col_idx]
            colors = ["#d62728" if lbl == "CIMig" else "#1f77b4" for lbl in labels]
            ax.bar(x, sub[metric], color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{DIRECTION_LABELS[direction]} — {'Cosine' if metric == 'cosine' else 'CrystalBLEU'}")
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_similarity_comparison-rq1.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rq3_boxplots(detail: pd.DataFrame, out_dir: Path) -> None:
    """fig_boxplot_comparison.pdf — per-project score distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    configs = [
        ("cimig", "cimig", "CIMig", "Set_Full", None),
        ("zero-shot", "gpt-4o", "GPT-4o (ZS)", "Set_Full", None),
        ("few-shot", "gpt-4o", "GPT-4o (FS)", "Set_Full", None),
        ("fine-tuned", "gemma3_12b", "Gemma FT", "Set_90_10", "Set_90_10"),
    ]

    for row_idx, direction in enumerate(["travis_to_gha", "gha_to_travis"]):
        for col_idx, metric_col in enumerate(["max_cosine_similarity", "max_crystal_bleu"]):
            ax = axes[row_idx, col_idx]
            data, labels = [], []
            for mode, model, label, set_name, fold_prefix in configs:
                q = detail[
                    (detail.prompt_mode == mode)
                    & (detail.migration_direction == direction)
                    & (detail.set_folder_name == set_name)
                ]
                if model != "cimig":
                    q = q[q.model_name == model]
                if fold_prefix:
                    q = q[q.fold_name.str.startswith(fold_prefix, na=False)]
                data.append(q[metric_col].values)
                labels.append(label)

            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.set_title(
                f"{DIRECTION_LABELS[direction]} — "
                f"{'Cosine' if 'cosine' in metric_col else 'CrystalBLEU'}"
            )
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_boxplot_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_improvement_heatmap(config: pd.DataFrame, out_dir: Path) -> None:
    """fig_improvement_heatmap.pdf — % improvement over CIMig baseline."""
    labels = [lbl for lbl in config.label.unique() if lbl != "CIMig"]
    directions = ["travis_to_gha", "gha_to_travis"]
    metrics = ["cosine", "bleu"]
    matrix = np.zeros((len(labels), len(directions) * len(metrics)))

    for j, direction in enumerate(directions):
        base = config[(config.label == "CIMig") & (config.direction == direction)].iloc[0]
        for k, metric in enumerate(metrics):
            col = j * len(metrics) + k
            for i, label in enumerate(labels):
                val = config[(config.label == label) & (config.direction == direction)][metric].iloc[0]
                base_val = base[metric]
                pct = ((val - base_val) / base_val * 100) if base_val else 0.0
                matrix[i, col] = pct

    col_labels = []
    for d in directions:
        for m in metrics:
            col_labels.append(f"{DIRECTION_LABELS[d]}\n{m.title()}")

    fig, ax = plt.subplots(figsize=(8, max(6, len(labels) * 0.35)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-20, vmax=350, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="% improvement over CIMig")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_improvement_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_linting_results(lint_csv: Path, out_dir: Path) -> None:
    """fig_linting_results.pdf — YAML validity and linter pass rates."""
    if not lint_csv.exists():
        print(f"Skipping linting figure; missing {lint_csv}")
        return

    df = pd.read_csv(lint_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, direction, title in zip(
        axes,
        ["travis_to_gha", "gha_to_travis"],
        ["Travis → GHA", "GHA → Travis"],
    ):
        sub = df[df.migration_direction == direction].copy()
        labels = sub["label"].tolist()
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, sub["yaml_valid_pct"], width, label="PyYAML valid")
        ax.bar(x + width / 2, sub["linter_pass_pct"], width, label="CI linter pass")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_linting_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from similarity CSVs.")
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--lint-csv",
        type=Path,
        default=PACKAGE_ROOT / "results" / "linting" / "linting_summary.csv",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, detail = load_tables(args.sim_dir)
    # fold-level aggregates for fine-tuned rows
    folds = pd.read_csv(args.sim_dir / "summary_across_folds.csv")
    summary = pd.concat([summary, folds], ignore_index=True, sort=False)

    plot_rq1_comparison(summary, args.output_dir)
    plot_finetuning_folds(detail, args.output_dir)
    config = _config_means(summary)
    plot_rq3_bars(config, args.output_dir)
    plot_rq3_boxplots(detail, args.output_dir)
    plot_improvement_heatmap(config, args.output_dir)
    plot_linting_results(args.lint_csv, args.output_dir)

    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
