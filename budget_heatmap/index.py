import re
from pathlib import Path as FilePath
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.facecolor": "#f7f9fc",
        "figure.facecolor": "white",
        "axes.edgecolor": "#c5ccd6",
        "grid.color": "#dce1ec",
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
    }
)

BASE_DIR = FilePath(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "budget_heatmap.csv"
OUTPUT_PNG = BASE_DIR / "budget_heatmap.png"
OUTPUT_PDF = BASE_DIR / "budget_heatmap.pdf"

BUDGET_ORDER = ["CoT", "1", "3", "5", "7", "Auto"]
ORDER_MAP = {label: idx for idx, label in enumerate(BUDGET_ORDER)}
FIG_HEIGHT = 5.2
BASE_WIDTH = 4.8
SHOW_TOKENS = True
MAX_ALLOWED_ERRORS: Optional[int] = None


def format_dataset_name(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return None
    cleaned = name.strip()
    if "/" in cleaned:
        cleaned = cleaned.split("/")[-1]
    return cleaned.replace("_", "/")


def format_model_name(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return None
    cleaned = name.strip()
    if cleaned.lower().startswith("sculpt-ai"):
        return "GIM-4B"
    if "/" in cleaned:
        cleaned = cleaned.split("/")[-1]
    return cleaned


def get_column_name(df, *candidates):
    lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate is None:
            continue
        real = lookup.get(candidate.lower())
        if real:
            return real
    return None


def pick_first_available(df, candidates):
    for candidate in candidates:
        col = get_column_name(df, candidate)
        if col:
            return col
    return None


def strip_percentage(series):
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


def extract_from_args(text, key, pattern):
    if not isinstance(text, str):
        return None
    match = re.search(pattern.format(key=key), text)
    if match:
        return match.group(1)
    return None


def determine_dataset(row, args_col: Optional[str], filename_col: Optional[str]):
    args_text = row[args_col] if args_col else ""
    if isinstance(args_text, str):
        dataset_match = re.search(r"'dataset':\s*\{[^}]*'path':\s*'([^']+)'", args_text)
        if dataset_match:
            return dataset_match.group(1)

    filename = row[filename_col] if filename_col else ""
    if isinstance(filename, str) and filename:
        match = re.search(r"_(?:[a-z]+_[a-z0-9]+)_", filename)
        if match:
            token = match.group(0).strip("_")
            if token:
                return token
        parts = filename.split("_")
        if len(parts) >= 3:
            candidate = "_".join(parts[2:-1])
            if candidate:
                return candidate
        return filename

    return None


def detect_budget(row, reason_col, args_col, filename_col):
    args_text = row[args_col] if args_col else ""
    filename_text = row[filename_col] if filename_col else ""
    combined_text = f"{args_text} {filename_text}".lower()

    auto_flag = extract_from_args(
        args_text, "auto_budget", r"'auto_budget':\s*(True|False)"
    )
    auto_requested = (auto_flag or "").lower() == "true" or "--auto_budget" in combined_text

    reason_val = None
    if reason_col and not pd.isna(row[reason_col]):
        try:
            reason_val = float(row[reason_col])
        except (TypeError, ValueError):
            reason_val = None

    if auto_requested:
        return "Auto"

    if reason_val is not None:
        if abs(reason_val) < 1e-3:
            return "CoT"
        for target in (1, 3, 5, 7):
            if abs(reason_val - target) <= 0.2:
                return str(target)
        return "Auto"

    if "cot" in combined_text:
        return "CoT"

    for label in ("1", "3", "5", "7"):
        if re.search(rf"(budget|reason)[^\d]{{0,3}}{label}\b|[_-]{label}(?:_|-|\b)", combined_text):
            return label

    return None


def determine_model(row, model_col, args_col, filename_col):
    if model_col:
        val = row[model_col]
        if isinstance(val, str) and val.strip():
            return val.strip()

    args_text = row[args_col] if args_col else ""
    model_name = extract_from_args(args_text, "model_name", r"'model_name':\s*'([^']+)'")
    if model_name:
        return model_name

    filename = row[filename_col] if filename_col else ""
    if isinstance(filename, str) and filename.strip():
        match = re.match(r"([^_]+_[^_]+)", filename.strip())
        if match:
            return match.group(1)
        return filename.strip()

    return None


def load_dataframe(path):
    """Load CSV with a utf-8 preference and graceful fallback."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def main():
    df = load_dataframe(INPUT_PATH)

    accuracy_col = pick_first_available(df, ["calibrated_accuracy", "accuracy"])
    tokens_col = None
    if SHOW_TOKENS:
        tokens_col = pick_first_available(df, ["avg_response_tokens", "avg_total_tokens"])
    errors_col = get_column_name(df, "errors")
    reason_col = get_column_name(df, "avg_reason_budget")
    args_col = get_column_name(df, "args")
    filename_col = get_column_name(df, "filename")
    model_col = get_column_name(df, "model", "Model")
    dataset_col = filename_col  # base column for fallback lookups

    if accuracy_col is None:
        raise ValueError("Missing required columns for accuracy.")
    if SHOW_TOKENS and tokens_col is None:
        raise ValueError("Missing required columns for token counts.")

    df["accuracy_num"] = strip_percentage(df[accuracy_col])
    if SHOW_TOKENS:
        df["token_num"] = pd.to_numeric(df[tokens_col], errors="coerce")

    if errors_col is not None and MAX_ALLOWED_ERRORS is not None:
        df = df[pd.to_numeric(df[errors_col], errors="coerce") <= MAX_ALLOWED_ERRORS]

    drop_cols = ["accuracy_num"]
    if SHOW_TOKENS:
        drop_cols.append("token_num")
    df = df.dropna(subset=drop_cols)

    df["model_label"] = df.apply(
        determine_model,
        axis=1,
        args=(model_col, args_col, filename_col),
    )
    df["dataset_label"] = df.apply(
        determine_dataset,
        axis=1,
        args=(args_col, dataset_col),
    )
    df["budget_label"] = df.apply(
        detect_budget,
        axis=1,
        args=(reason_col, args_col, filename_col),
    )

    df = df.dropna(subset=["model_label", "budget_label", "dataset_label"])
    df["model_label"] = df["model_label"].apply(format_model_name)
    df["dataset_label"] = df["dataset_label"].apply(format_dataset_name)
    df = df[df["budget_label"].isin(BUDGET_ORDER)]

    if df.empty:
        raise ValueError("No valid rows left after filtering; check the input data.")

    agg_map = {"accuracy_num": "mean"}
    if SHOW_TOKENS:
        agg_map["token_num"] = "mean"
    summary = (
        df.groupby(["dataset_label", "model_label", "budget_label"], as_index=False)
        .agg(agg_map)
    )
    summary["order_idx"] = summary["budget_label"].map(ORDER_MAP)
    summary = summary.dropna(subset=["order_idx"])
    if summary.empty:
        raise ValueError("Insufficient data after aggregating by dataset and budget.")

    summary = summary.dropna(subset=["dataset_label", "model_label"])
    datasets = sorted(summary["dataset_label"].unique())
    ncols = len(datasets)
    global_model_order = (
        summary.groupby("model_label")["accuracy_num"].mean().sort_values(ascending=False).index.tolist()
    )
    if not global_model_order:
        raise ValueError("No models available for visualization.")

    fig_width = max(9.5, 4.3 * ncols + 0.8)
    fig_height = max(4.2, 2.0 + 0.5 * len(global_model_order))
    fig = plt.figure(figsize=(fig_width, fig_height))
    width_ratios = [1.0] * ncols + [0.08]
    grid = fig.add_gridspec(1, ncols + 1, width_ratios=width_ratios, wspace=0.22)
    axes = [fig.add_subplot(grid[0, i]) for i in range(ncols)]
    gradient_ax = fig.add_subplot(grid[0, -1])

    cmap = LinearSegmentedColormap.from_list(
        "soft_accuracy",
        ["#f8fbff", "#dfe9fb", "#b9d2f3", "#91b6e4", "#5f8fc8"],
    )
    text_outline = [path_effects.Stroke(linewidth=1.2, foreground="black", alpha=0.35), path_effects.Normal()]
    token_outline = [path_effects.Stroke(linewidth=0.8, foreground="white", alpha=0.55), path_effects.Normal()]

    for ax_idx, (ax, dataset_name) in enumerate(zip(axes, datasets)):
        dataset_df = summary[summary["dataset_label"] == dataset_name]
        heat = np.full((len(global_model_order), len(BUDGET_ORDER)), np.nan)
        token_grid = np.full_like(heat, np.nan)

        for m_idx, model_name in enumerate(global_model_order):
            model_rows = dataset_df[dataset_df["model_label"] == model_name]
            if model_rows.empty:
                continue
            for _, row in model_rows.iterrows():
                b_idx = ORDER_MAP.get(row["budget_label"])
                if b_idx is None:
                    continue
                heat[m_idx, b_idx] = row["accuracy_num"]
                token_val = row.get("token_num") if "token_num" in row else None
                if SHOW_TOKENS and pd.notna(token_val):
                    token_grid[m_idx, b_idx] = token_val

        if np.isnan(heat).all():
            ax.axis("off")
            continue

        local_min = np.nanmin(heat)
        local_max = np.nanmax(heat)
        if abs(local_max - local_min) < 1e-6:
            local_max = local_min + 1e-6
        local_span = local_max - local_min

        ax.axvspan(-0.5, 0.5, color="#f1f3fb", alpha=0.8, zorder=0)
        ax.imshow(heat, aspect="auto", cmap=cmap, vmin=local_min, vmax=local_max, zorder=1)
        ax.set_xticks(range(len(BUDGET_ORDER)))
        ax.set_xticklabels(BUDGET_ORDER, fontsize=10, fontweight="medium")
        ax.set_yticks(range(len(global_model_order)))
        if ax_idx == 0:
            ax.set_yticklabels(
                [format_model_name(m) for m in global_model_order],
                fontsize=8.6,
                fontweight="medium",
                ha="right",
            )
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", pad=15)
        # === Paper Terminology Alignment ===
        display_name = dataset_name
        if isinstance(dataset_name, str):
            normalized = dataset_name.lower()
            replacements = {
                "medmcqa": "MedMCQA",
                "qasc": "QASC",
            }
            display_name = replacements.get(normalized, dataset_name)
        ax.set_xlabel(
            "Reasoning Budget (CoT vs. GIM)",
            fontsize=10,
            fontweight="normal",
            labelpad=14,
        )
        ax.set_title(display_name, fontsize=12, fontweight="semibold", pad=10)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_facecolor("#f4f6fb")
        ax.set_xlim(-0.5, len(BUDGET_ORDER) - 0.5)
        ax.set_ylim(len(global_model_order) - 0.5, -0.5)
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(-0.5, len(BUDGET_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(global_model_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.axvline(0.5, color="#9aa5c4", linewidth=0.9, linestyle="--")
        bracket_top = -0.04
        bracket_bottom = -0.10
        verts = [
            (0.5, bracket_top),
            (0.5, bracket_bottom),
            (len(BUDGET_ORDER) - 0.5, bracket_bottom),
            (len(BUDGET_ORDER) - 0.5, bracket_top),
        ]
        path = Path(verts, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
        bracket = PathPatch(
            path,
            transform=ax.get_xaxis_transform(),
            linewidth=1.2,
            color="#9aa5c4",
            fill=False,
            capstyle="round",
            joinstyle="round",
            clip_on=False,
        )
        ax.add_patch(bracket)

        for r_idx in range(len(global_model_order)):
            for c_idx in range(len(BUDGET_ORDER)):
                acc_val = heat[r_idx, c_idx]
                if np.isnan(acc_val):
                    continue
                norm_val = (acc_val - local_min) / local_span
                text_color = "#fafafa" if norm_val >= 0.6 else "#1f2a44"
                ax.text(
                    c_idx,
                    r_idx - 0.15,
                    f"{acc_val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="semibold",
                    color=text_color,
                    path_effects=text_outline,
                )
                if SHOW_TOKENS and not np.isnan(token_grid[r_idx, c_idx]):
                    ax.text(
                        c_idx,
                        r_idx + 0.22,
                        f"{token_grid[r_idx, c_idx]:.0f} tok",
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color="#f6f8fc" if norm_val >= 0.75 else "#223040",
                        alpha=0.88,
                        path_effects=token_outline,
                    )
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient_ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower")
    gradient_ax.set_xticks([])
    gradient_ax.set_yticks([])
    gradient_ax.set_facecolor("white")
    for spine in gradient_ax.spines.values():
        spine.set_visible(False)
    gradient_ax.annotate(
        "",
        xy=(0.5, 0.92),
        xytext=(0.5, 0.08),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#4a5268", lw=0.9),
        annotation_clip=False,
    )
    # === Paper Terminology Alignment ===
    gradient_ax.text(
        1.25,
        0.5,
        "Accuracy (%)",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="semibold",
        color="#4a5268",
        rotation=90,
        rotation_mode="anchor",
        transform=gradient_ax.transAxes,
    )

    if SHOW_TOKENS:
        # === Paper Terminology Alignment ===
        fig.text(
            0.96,
            0.08,
            "Values marked 'tok' report mean response tokens.",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="medium",
            color="#4a5268",
        )

    fig.subplots_adjust(left=0.22, right=0.965, top=0.94, bottom=0.21, wspace=0.16)
    heat_pos = axes[0].get_position()
    grad_pos = gradient_ax.get_position()
    gradient_ax.set_position([grad_pos.x0, heat_pos.y0, grad_pos.width, heat_pos.height])

    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_PDF, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
