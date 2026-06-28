"""
plot_results_met2.py
====================
Reads results_met2/GP/ and produces three sets of plots:

Plot 1 — Accuracy heatmap  (n_demes x migration_rate) per dataset
Plot 2 — Behavioural diversity heatmap (same layout)
Plot 3 — Generational curves per dataset (averaged over 30 runs ± std):
          • best_train_fitness
          • best_ind_nodes
          • behavioural_diversity
          Each plot has 5 curves: tournament baseline + 4 best deme configs

Usage (from inside met2/ folder):
    python codes/plot_results_met2.py
Outputs saved to results_met2/GP/plots/
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_ROOT = "results_met2/GP"
PLOTS_DIR    = "results_met2/GP/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASETS   = ["wine", "iris", "australian", "pima", "heart"]
N_DEMES    = [5, 10, 15, 20]
MR_PCTS    = [2, 5, 10, 15]
GENERATIONS = 50

# Colour palette for generational curves
COLORS = {
    "tournament":     "#e74c3c",   # red
    "demes_5_mr2":   "#1a6faf",
    "demes_5_mr5":   "#2196F3",
    "demes_10_mr2":  "#2e7d32",
    "demes_10_mr5":  "#4CAF50",
    "demes_10_mr10": "#ff9800",
    "demes_10_mr15": "#ff5722",
    "demes_15_mr2":  "#7b1fa2",
    "demes_15_mr5":  "#9c27b0",
    "demes_20_mr2":  "#00695c",
    "demes_20_mr5":  "#009688",
}
FALLBACK_COLOR = "#888888"


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_final_metrics(folder):
    """Return (mean_acc, std_acc, mean_bd, std_bd) from all CSVs in folder."""
    accs, bds = [], []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            df = pd.read_csv(f, sep='\t')
            acc = df["accuracy_test"].dropna()
            bd  = df["behavioural_diversity"].dropna()
            if len(acc) > 0: accs.append(float(acc.iloc[-1]))
            if len(bd)  > 0: bds.append(float(bd.iloc[-1]))
        except Exception:
            pass
    if not accs:
        return np.nan, np.nan, np.nan, np.nan
    return (np.mean(accs), np.std(accs),
            np.mean(bds) if bds else np.nan,
            np.std(bds)  if bds else np.nan)


def load_generational(folder, metric, n_gen=GENERATIONS):
    """
    Returns (mean_curve, std_curve) arrays of length n_gen+1,
    averaged over all CSV runs in folder.
    """
    curves = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            df = pd.read_csv(f, sep='\t')
            if metric in df.columns:
                vals = df[metric].values[:n_gen + 1]
                if len(vals) == n_gen + 1:
                    curves.append(vals.astype(float))
        except Exception:
            pass
    if not curves:
        return None, None
    arr = np.array(curves)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 & 2 — Heatmaps: accuracy and behavioural diversity
# ═══════════════════════════════════════════════════════════════════════════════
def plot_heatmaps(dataset):
    acc_grid = np.full((len(N_DEMES), len(MR_PCTS)), np.nan)
    bd_grid  = np.full((len(N_DEMES), len(MR_PCTS)), np.nan)

    for i, nd in enumerate(N_DEMES):
        for j, mr in enumerate(MR_PCTS):
            folder = os.path.join(RESULTS_ROOT, dataset, f"demes_{nd}_mr{mr}")
            if not os.path.exists(folder):
                continue
            mean_acc, _, mean_bd, _ = load_final_metrics(folder)
            acc_grid[i, j] = mean_acc
            bd_grid[i, j]  = mean_bd

    # Tournament baseline value for reference
    tourn_folder = os.path.join(RESULTS_ROOT, dataset, "tournament")
    tourn_acc = np.nan
    if os.path.exists(tourn_folder):
        tourn_acc, _, _, _ = load_final_metrics(tourn_folder)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset.capitalize()} — Accuracy & Behavioural Diversity Heatmaps\n"
                 f"(Tournament baseline accuracy = {tourn_acc:.4f})",
                 fontsize=13, fontweight='bold')

    mr_labels = [f"{m}%" for m in MR_PCTS]
    nd_labels = [str(n) for n in N_DEMES]

    for ax, grid, title, cmap in [
        (axes[0], acc_grid, "Mean Test Accuracy",          "YlGn"),
        (axes[1], bd_grid,  "Mean Behavioural Diversity",  "YlOrRd"),
    ]:
        # Mask NaN cells
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap=cmap, aspect='auto',
                       vmin=np.nanmin(grid) - 0.005 if not np.all(np.isnan(grid)) else 0,
                       vmax=np.nanmax(grid) + 0.005 if not np.all(np.isnan(grid)) else 1)
        plt.colorbar(im, ax=ax, shrink=0.85)

        ax.set_xticks(range(len(MR_PCTS)));  ax.set_xticklabels(mr_labels)
        ax.set_yticks(range(len(N_DEMES)));  ax.set_yticklabels(nd_labels)
        ax.set_xlabel("Migration Rate", fontsize=10)
        ax.set_ylabel("Number of Demes", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Annotate cells
        for i in range(len(N_DEMES)):
            for j in range(len(MR_PCTS)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha='center', va='center',
                            fontsize=9, fontweight='bold',
                            color='white' if val < (np.nanmax(grid) * 0.85) else 'black')

        # Mark best cell with a border
        if not np.all(np.isnan(grid)):
            best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
            ax.add_patch(plt.Rectangle(
                (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=3, label='Best'
            ))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{dataset}_heatmaps.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Generational curves
# ═══════════════════════════════════════════════════════════════════════════════
def get_top_deme_configs(dataset, n_top=4):
    """Return the top-n deme configs by mean accuracy for a dataset."""
    scores = []
    for nd in N_DEMES:
        for mr in MR_PCTS:
            folder = os.path.join(RESULTS_ROOT, dataset, f"demes_{nd}_mr{mr}")
            if not os.path.exists(folder):
                continue
            mean_acc, _, _, _ = load_final_metrics(folder)
            if not np.isnan(mean_acc):
                scores.append((mean_acc, f"demes_{nd}_mr{mr}", nd, mr))
    scores.sort(reverse=True)
    return scores[:n_top]


def plot_generational_curves(dataset):
    top_configs = get_top_deme_configs(dataset, n_top=4)
    gens = np.arange(GENERATIONS + 1)

    metrics = [
        ("best_train_fitness",   "Best Train Fitness (RMSE)",    True),
        ("best_ind_nodes",       "Best Individual Nodes",         False),
        ("behavioural_diversity","Behavioural Diversity",         False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset.capitalize()} — Generational Curves (mean ± std over 30 runs)",
                 fontsize=13, fontweight='bold')

    for ax, (metric, ylabel, invert) in zip(axes, metrics):
        # Tournament baseline
        tourn_folder = os.path.join(RESULTS_ROOT, dataset, "tournament")
        if os.path.exists(tourn_folder):
            mean_c, std_c = load_generational(tourn_folder, metric)
            if mean_c is not None:
                ax.plot(gens, mean_c, color=COLORS["tournament"],
                        linewidth=2.5, label="Tournament (baseline)", zorder=5)
                ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                                color=COLORS["tournament"], alpha=0.12)

        # Top deme configs
        for rank, (mean_acc, config_key, nd, mr) in enumerate(top_configs):
            folder = os.path.join(RESULTS_ROOT, dataset, config_key)
            mean_c, std_c = load_generational(folder, metric)
            if mean_c is None:
                continue
            label  = f"Demes n={nd}, mr={mr}%  (acc={mean_acc:.4f})"
            color  = COLORS.get(config_key, FALLBACK_COLOR)
            ls     = ['-', '--', '-.', ':'][rank % 4]
            ax.plot(gens, mean_c, color=color, linewidth=1.8,
                    linestyle=ls, label=label)
            ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.08)

        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.set_xlim(0, GENERATIONS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(fontsize=7.5, framealpha=0.8, loc='best')
        if invert:
            ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{dataset}_generational.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Accuracy bar chart across all datasets (best deme config per dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_overall_accuracy():
    tourn_accs  = []
    best_accs   = []
    best_labels = []
    ds_labels   = []

    for dataset in DATASETS:
        # Tournament
        tourn_folder = os.path.join(RESULTS_ROOT, dataset, "tournament")
        if not os.path.exists(tourn_folder):
            continue
        t_acc, t_std, _, _ = load_final_metrics(tourn_folder)
        if np.isnan(t_acc):
            continue

        # Best deme config
        top = get_top_deme_configs(dataset, n_top=1)
        if not top:
            continue
        b_acc, b_key, b_nd, b_mr = top[0]
        _, b_std, _, _ = load_final_metrics(
            os.path.join(RESULTS_ROOT, dataset, b_key))

        tourn_accs.append((t_acc, t_std))
        best_accs.append((b_acc, b_std))
        best_labels.append(f"n={b_nd}, mr={b_mr}%")
        ds_labels.append(dataset.capitalize())

    if not ds_labels:
        print("  No data yet for overall accuracy plot — skipping.")
        return

    x = np.arange(len(ds_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - w/2, [v[0] for v in tourn_accs], w,
                   yerr=[v[1] for v in tourn_accs],
                   label="Tournament (baseline)", color="#e74c3c",
                   capsize=5, error_kw={"elinewidth":1.5})
    bars2 = ax.bar(x + w/2, [v[0] for v in best_accs], w,
                   yerr=[v[1] for v in best_accs],
                   label="Best Demes Config", color="#27ae60",
                   capsize=5, error_kw={"elinewidth":1.5})

    # Annotate bars
    for bar, (acc, _), bl in zip(bars2, best_accs, best_labels):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{acc:.4f}\n({bl})",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for bar, (acc, _) in zip(bars1, tourn_accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{acc:.4f}",
                ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(ds_labels, fontsize=11)
    ax.set_ylabel("Mean Test Accuracy (30 runs)", fontsize=10)
    ax.set_title("Tournament vs Best Demes Config — All Datasets", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ymin = max(0, min([v[0] for v in tourn_accs + best_accs]) - 0.05)
    ax.set_ylim(ymin, 1.02)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "overall_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")
print("─" * 50)

for ds in DATASETS:
    ds_path = os.path.join(RESULTS_ROOT, ds)
    if not os.path.exists(ds_path):
        print(f"  Skipping {ds} — no results yet")
        continue
    print(f"\n{ds.upper()}")
    plot_heatmaps(ds)
    plot_generational_curves(ds)

print("\nOVERALL")
plot_overall_accuracy()

print(f"\nAll plots saved to {PLOTS_DIR}/")
