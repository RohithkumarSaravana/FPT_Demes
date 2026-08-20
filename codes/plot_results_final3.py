"""
plot_results_final3.py  —  Result figures for the island-model GP experiments
==============================================================================
Generates the full set of figures used to analyse the island-model (demes)
GP results against the tournament-selection baseline:

  - Profile plots showing mean final behavioural diversity (BD) as a
    function of migration rate, number of demes, and migration frequency,
    with per-dataset markers and collision-avoiding data labels.
  - Generational curves (best train fitness, best individual node count,
    behavioural diversity) comparing the tournament baseline against a
    set of representative island-model configurations, plotted as
    mean ± std across runs with the legend shown once per figure.
  - Per-dataset heatmaps of mean test accuracy and mean final BD across
    the (number of demes) x (migration rate) grid for each migration
    frequency, with the accuracy colour scale centred on the tournament
    baseline, a shared global colour scale for BD across frequencies,
    and ▲/▼ markers indicating statistically significant differences
    from the baseline (from the Wilcoxon test results).

Usage (from inside the project root folder):
    python codes/plot_results_final3.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

# ── Paths and grid ─────────────────────────────────────────────────────────────
RESULTS_ROOT = "results_final/GP"
PLOTS_DIR    = "results_final/GP/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASETS    = ["wine", "iris", "australian", "pima", "heartDisease"]
DS_LABELS   = {"wine":"Wine","iris":"Iris","australian":"Australian",
                "pima":"Pima","heartDisease":"Heart Disease"}
N_DEMES_ALL = [5, 10, 15]
MR_PCTS     = [2, 5, 10, 15]
FREQS       = [1, 5, 10, 25]
GENERATIONS = 50

DS_COLORS  = {"wine":"#e74c3c","iris":"#2980b9","australian":"#27ae60",
               "pima":"#e67e22","heartDisease":"#8e44ad"}
DS_MARKERS = {"wine":"o","iris":"s","australian":"^","pima":"D","heartDisease":"v"}
COL_TOURN  = "#2c3e50"

REPRESENTATIVE_CONFIGS = [
    ("demes_10_mr5_freq1",  "#e74c3c", "n=10, mr=5%, freq=1",  "-"),
    ("demes_10_mr5_freq5",  "#2980b9", "n=10, mr=5%, freq=5",  "--"),
    ("demes_10_mr5_freq10", "#27ae60", "n=10, mr=5%, freq=10", "-."),
    ("demes_10_mr5_freq25", "#e67e22", "n=10, mr=5%, freq=25", ":"),
]

# Tournament baselines (for heatmap centring and titles)
TOURN_ACC = {"wine":0.9111,"iris":0.9553,"australian":0.8581,
              "pima":0.7542,"heartDisease":0.8182}
TOURN_BD  = {"wine":0.1236,"iris":0.0622,"australian":0.1117,
              "pima":0.3083,"heartDisease":0.2192}

# Load Wilcoxon results for significance markers on heatmaps
_wil_cache = None
def get_wilcoxon():
    """Load and cache wilcoxon_results.csv (from wilcoxon_analysis_final.py)
    for use by get_sig(). Returns an empty DataFrame if the file is missing."""
    global _wil_cache
    if _wil_cache is None:
        try:
            _wil_cache = pd.read_csv(os.path.join(RESULTS_ROOT, "wilcoxon_results.csv"))
        except Exception:
            _wil_cache = pd.DataFrame()
    return _wil_cache

def get_sig(dataset, nd, mr, freq, metric):
    """Return '+', '=', or '-' for (dataset, config, metric) from Wilcoxon CSV."""
    wil = get_wilcoxon()
    if wil.empty:
        return '='
    config = f"demes_{nd}_mr{mr}_freq{freq}"
    col    = "sig_acc" if metric == "acc" else "sig_bd"
    row    = wil[(wil['dataset'] == dataset) & (wil['config'] == config)]
    if row.empty:
        return '='
    return row[col].values[0]

# ── Loaders ────────────────────────────────────────────────────────────────────
def load_bd_final(folder):
    """Mean final-generation behavioural diversity across all run CSVs in folder."""
    bds = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            df  = pd.read_csv(f, sep='\t')
            col = df["behavioural_diversity"].dropna()
            if len(col) > 0:
                bds.append(float(col.iloc[-1]))
        except Exception:
            pass
    return float(np.mean(bds)) if bds else float('nan')

def load_acc_final(folder):
    """Mean final-generation test accuracy across all run CSVs in folder."""
    accs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            df  = pd.read_csv(f, sep='\t')
            col = df["accuracy_test"].dropna()
            if len(col) > 0:
                accs.append(float(col.iloc[-1]))
        except Exception:
            pass
    return float(np.mean(accs)) if accs else float('nan')

def load_generational(folder, metric):
    """
    Load a per-generation metric curve from every run CSV in folder
    (keeping only runs with the full GENERATIONS+1 rows) and return its
    mean and std across runs.

    Parameters:
        folder -- directory containing per-run result CSVs.
        metric -- column name to extract (e.g. 'best_train_fitness').

    Returns:
        Tuple (mean_curve, std_curve), each a NumPy array of length
        GENERATIONS+1, or (None, None) if no usable runs were found.
    """
    curves = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            df = pd.read_csv(f, sep='\t')
            if metric in df.columns:
                v = df[metric].values[:GENERATIONS + 1].astype(float)
                if len(v) == GENERATIONS + 1:
                    curves.append(v)
        except Exception:
            pass
    if not curves:
        return None, None
    arr = np.array(curves)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

# ── Compute global BD range per dataset for consistent heatmap scale ───────────
def get_global_bd_range(dataset):
    """Returns (bd_min, bd_max) across all deme configs for this dataset."""
    vals = []
    for nd in N_DEMES_ALL:
        for mr in MR_PCTS:
            for freq in FREQS:
                folder = os.path.join(RESULTS_ROOT, dataset,
                                      f"demes_{nd}_mr{mr}_freq{freq}")
                bd = load_bd_final(folder)
                if not np.isnan(bd):
                    vals.append(bd)
    if not vals:
        return 0, 1
    return min(vals), max(vals)

# ── Smart label offsets for profile plots to avoid collision ─────────────────
def _annotate_no_collision(ax, x_pos, y_vals, colors, labels, default_offset=10):
    """
    Annotate each point; apply alternating vertical offsets when two
    consecutive datasets at the same x position are too close.
    """
    # Collect all (xi, yi, color, label) per x position
    per_x = {xi: [] for xi in x_pos}
    for ds_idx, (yv, col) in enumerate(zip(y_vals, colors)):
        for xi, yi in zip(x_pos, yv):
            if not np.isnan(yi):
                per_x[xi].append((yi, col))

    # For each dataset, build per-point offsets
    # Simple strategy: sort by y value at each x; assign alternating up/down
    for xi in x_pos:
        pts = sorted(per_x[xi], key=lambda t: t[0])
        n   = len(pts)
        for rank, (yi, col) in enumerate(pts):
            # alternate above / below based on rank
            if n <= 2:
                oy = default_offset if rank == 0 else -14
            else:
                oy = default_offset if rank % 2 == 0 else -14
            ax.annotate(f"{yi:.3f}", (xi, yi),
                        textcoords="offset points",
                        xytext=(0, oy), ha='center',
                        fontsize=8, color=col)


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Profile: BD vs Migration Rate
# ════════════════════════════════════════════════════════════════════════════════
def plot_profile_bd_vs_mr():
    """Plot mean final behavioural diversity vs migration rate, one line
    per dataset (averaged over all n_demes and migration frequencies),
    with the tournament baseline as the leftmost point. Saves
    profile_BD_vs_migration_rate.png."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x_labels = ["Tournament\n(baseline)", "2%", "5%", "10%", "15%"]
    x_pos    = list(range(len(x_labels)))

    all_y    = []
    all_cols = []
    for ds in DATASETS:
        color = DS_COLORS[ds]
        y_vals = []
        t_folder = os.path.join(RESULTS_ROOT, ds, "tournament")
        y_vals.append(load_bd_final(t_folder))
        for mr in MR_PCTS:
            bds = []
            for nd in N_DEMES_ALL:
                for freq in FREQS:
                    folder = os.path.join(RESULTS_ROOT, ds,
                                          f"demes_{nd}_mr{mr}_freq{freq}")
                    bd = load_bd_final(folder)
                    if not np.isnan(bd):
                        bds.append(bd)
            y_vals.append(float(np.mean(bds)) if bds else float('nan'))

        if sum(1 for v in y_vals if not np.isnan(v)) < 3:
            continue
        ax.plot(x_pos, y_vals, color=color, marker=DS_MARKERS[ds],
                lw=2, markersize=8, label=DS_LABELS[ds])
        all_y.append(y_vals)
        all_cols.append(color)

    _annotate_no_collision(ax, x_pos, all_y, all_cols, DATASETS)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel(
        "Migration Rate\n"
        "(BD averaged across all n_demes ∈ {5,10,15} "
        "and all migration frequencies ∈ {1,5,10,25})",
        fontsize=10)
    ax.set_ylabel("Mean Final Behavioural Diversity", fontsize=11)
    ax.set_title("Behavioural Diversity vs Migration Rate",
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.85, loc='best')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "profile_BD_vs_migration_rate.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Profile: BD vs Number of Demes
# ════════════════════════════════════════════════════════════════════════════════
def plot_profile_bd_vs_ndemes():
    """Plot mean final behavioural diversity vs number of demes, one line
    per dataset (averaged over all migration rates and frequencies), with
    a horizontal tournament-baseline reference line per dataset. Saves
    profile_BD_vs_n_demes.png."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos    = list(range(len(N_DEMES_ALL)))
    x_labels = [str(n) for n in N_DEMES_ALL]

    all_y = []; all_cols = []
    for ds in DATASETS:
        color  = DS_COLORS[ds]
        y_vals = []
        for nd in N_DEMES_ALL:
            bds = []
            for mr in MR_PCTS:
                for freq in FREQS:
                    folder = os.path.join(RESULTS_ROOT, ds,
                                          f"demes_{nd}_mr{mr}_freq{freq}")
                    bd = load_bd_final(folder)
                    if not np.isnan(bd):
                        bds.append(bd)
            y_vals.append(float(np.mean(bds)) if bds else float('nan'))

        if sum(1 for v in y_vals if not np.isnan(v)) < 2:
            continue
        ax.plot(x_pos, y_vals, color=color, marker=DS_MARKERS[ds],
                lw=2, markersize=9, label=DS_LABELS[ds])
        all_y.append(y_vals); all_cols.append(color)

        # horizontal tournament reference line
        t_bd = TOURN_BD.get(ds, float('nan'))
        if not np.isnan(t_bd):
            ax.axhline(t_bd, color=color, lw=0.8, linestyle=':', alpha=0.4)

    _annotate_no_collision(ax, x_pos, all_y, all_cols, DATASETS)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel(
        "Number of Demes\n"
        "(BD averaged across all migration rates ∈ {2%,5%,10%,15%} "
        "and all migration frequencies ∈ {1,5,10,25})",
        fontsize=10)
    ax.set_ylabel("Mean Final Behavioural Diversity", fontsize=11)
    ax.set_title("Behavioural Diversity vs Number of Demes",
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.85, loc='best')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "profile_BD_vs_n_demes.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Profile: BD vs Migration Frequency
# ════════════════════════════════════════════════════════════════════════════════
def plot_profile_bd_vs_freq():
    """Plot mean final behavioural diversity vs migration frequency, one
    line per dataset (averaged over all n_demes and migration rates).
    Saves profile_BD_vs_migration_frequency.png."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x_pos    = list(range(len(FREQS)))
    x_labels = [str(f) for f in FREQS]

    all_y = []; all_cols = []
    for ds in DATASETS:
        color  = DS_COLORS[ds]
        y_vals = []
        for freq in FREQS:
            bds = []
            for nd in N_DEMES_ALL:
                for mr in MR_PCTS:
                    folder = os.path.join(RESULTS_ROOT, ds,
                                          f"demes_{nd}_mr{mr}_freq{freq}")
                    bd = load_bd_final(folder)
                    if not np.isnan(bd):
                        bds.append(bd)
            y_vals.append(float(np.mean(bds)) if bds else float('nan'))

        if sum(1 for v in y_vals if not np.isnan(v)) < 2:
            continue
        ax.plot(x_pos, y_vals, color=color, marker=DS_MARKERS[ds],
                lw=2, markersize=8, label=DS_LABELS[ds])
        all_y.append(y_vals); all_cols.append(color)

    _annotate_no_collision(ax, x_pos, all_y, all_cols, DATASETS)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel(
        "Migration Frequency (every N generations)\n"
        "(BD averaged across all n_demes ∈ {5,10,15} "
        "and all migration rates ∈ {2%,5%,10%,15%})",
        fontsize=10)
    ax.set_ylabel("Mean Final Behavioural Diversity", fontsize=11)
    ax.set_title("Behavioural Diversity vs Migration Frequency",
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.85, loc='best')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)
    ax.text(0.98, 0.02, "Higher value = less frequent migration",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "profile_BD_vs_migration_frequency.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURES 4–8 — Generational curves
# Node count panel is clipped at zero (no negative std bands); legend shown
# in the first subplot only to avoid repeating it across all three panels
# ════════════════════════════════════════════════════════════════════════════════
def plot_generational(dataset):
    """
    Plot generational curves (best train fitness, best individual node
    count, behavioural diversity; mean +/- std across runs) for one
    dataset, comparing the tournament baseline against the representative
    island-model configurations in REPRESENTATIVE_CONFIGS. Saves
    generational_<dataset>.png. No-ops (with a message) if the dataset
    has no tournament results.

    Parameters:
        dataset -- dataset name (key into RESULTS_ROOT/<dataset>).
    """
    t_folder = os.path.join(RESULTS_ROOT, dataset, "tournament")
    if not os.path.exists(t_folder):
        print(f"  Skipping {dataset} — no tournament results")
        return

    gens    = np.arange(GENERATIONS + 1)
    metrics = [
        ("best_train_fitness",    "Best Train Fitness (RMSE)"),
        ("best_ind_nodes",        "Best Individual Nodes"),
        ("behavioural_diversity", "Behavioural Diversity"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{DS_LABELS[dataset]} — Generational Curves  (mean ± std, 30 runs)\n"
        f"Tournament baseline vs island-model GP  "
        f"[n=10, mr=5%, varying migration frequency]",
        fontsize=11, fontweight='bold'
    )

    for ax_idx, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        show_legend = (ax_idx == 0)   # legend only in first panel

        mc, sc = load_generational(t_folder, metric)
        if mc is not None:
            lbl = "Tournament (baseline)" if show_legend else "_nolegend_"
            ax.plot(gens, mc, color=COL_TOURN, lw=2.5, linestyle='-',
                    label=lbl, zorder=5)
            lo = mc - sc
            hi = mc + sc
            if metric == "best_ind_nodes":  # clip node count at zero (cannot be negative)
                lo = np.maximum(lo, 0)
            ax.fill_between(gens, lo, hi, color=COL_TOURN, alpha=0.10)

        for config_key, color, label, ls in REPRESENTATIVE_CONFIGS:
            folder = os.path.join(RESULTS_ROOT, dataset, config_key)
            if not os.path.exists(folder):
                continue
            mc, sc = load_generational(folder, metric)
            if mc is None:
                continue
            lbl = label if show_legend else "_nolegend_"
            ax.plot(gens, mc, color=color, lw=1.8, linestyle=ls, label=lbl)
            lo = mc - sc
            hi = mc + sc
            if metric == "best_ind_nodes":  # clip node count at zero (cannot be negative)
                lo = np.maximum(lo, 0)
            ax.fill_between(gens, lo, hi, color=color, alpha=0.07)

        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.set_xlim(0, GENERATIONS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle='--', alpha=0.35)
        if show_legend:
            ax.legend(fontsize=8, framealpha=0.85, loc='best')

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"generational_{dataset}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# HEATMAPS
# Heatmap cells are marked with ▲/▼ significance indicators from the Wilcoxon
# test results. Accuracy uses a colour scale centred on the tournament baseline
# (TwoSlopeNorm); BD uses a global (dataset-level) scale shared across all
# migration-frequency figures so colours are comparable. Titles report both
# the baseline accuracy and baseline BD.
# ════════════════════════════════════════════════════════════════════════════════
def plot_heatmaps(dataset):
    """
    For one dataset, plot a pair of heatmaps (accuracy, BD) over the
    (number of demes) x (migration rate) grid, one figure per migration
    frequency. Saves <dataset>_heatmap_freq<freq>.png for each frequency
    in FREQS that has data.

    Parameters:
        dataset -- dataset name (key into RESULTS_ROOT/<dataset>,
                   TOURN_ACC, and TOURN_BD).
    """
    # Pre-compute global BD range for this dataset so the colour scale is
    # consistent across all migration-frequency figures
    bd_global_min, bd_global_max = get_global_bd_range(dataset)
    tourn_acc = TOURN_ACC[dataset]
    tourn_bd  = TOURN_BD[dataset]

    for freq in FREQS:
        acc_grid = np.full((len(N_DEMES_ALL), len(MR_PCTS)), np.nan)
        bd_grid  = np.full((len(N_DEMES_ALL), len(MR_PCTS)), np.nan)
        sig_acc  = [['' for _ in MR_PCTS] for _ in N_DEMES_ALL]
        sig_bd   = [['' for _ in MR_PCTS] for _ in N_DEMES_ALL]

        any_data = False
        for i, nd in enumerate(N_DEMES_ALL):
            for j, mr in enumerate(MR_PCTS):
                folder = os.path.join(RESULTS_ROOT, dataset,
                                      f"demes_{nd}_mr{mr}_freq{freq}")
                if not os.path.exists(folder):
                    continue
                acc = load_acc_final(folder)
                bd  = load_bd_final(folder)
                acc_grid[i, j] = acc
                bd_grid[i, j]  = bd
                sig_acc[i][j]  = get_sig(dataset, nd, mr, freq, "acc")
                sig_bd[i][j]   = get_sig(dataset, nd, mr, freq, "bd")
                any_data = True

        if not any_data:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{DS_LABELS[dataset]} — freq={freq}  "
            f"(migration every {freq} generation{'s' if freq > 1 else ''})\n"
            f"Tournament baseline: accuracy = {tourn_acc:.4f},  "
            f"BD = {tourn_bd:.4f}",
            fontsize=12, fontweight='bold')

        mr_labels = [f"{m}%" for m in MR_PCTS]
        nd_labels = [str(n) for n in N_DEMES_ALL]

        # ── Accuracy panel (colour scale centred on tournament baseline) ───
        ax = axes[0]
        valid_acc = acc_grid[~np.isnan(acc_grid)]
        if len(valid_acc) > 0:
            vmin = min(float(np.nanmin(acc_grid)), tourn_acc - 0.002)
            vmax = max(float(np.nanmax(acc_grid)), tourn_acc + 0.002)
            # Centre colormap on tournament baseline
            norm = TwoSlopeNorm(vmin=vmin, vcenter=tourn_acc, vmax=vmax)
            masked = np.ma.masked_invalid(acc_grid)
            im = ax.imshow(masked, cmap='RdYlGn', aspect='auto', norm=norm)
            plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xticks(range(len(MR_PCTS)));   ax.set_xticklabels(mr_labels)
        ax.set_yticks(range(len(N_DEMES_ALL))); ax.set_yticklabels(nd_labels)
        ax.set_xlabel("Migration Rate", fontsize=10)
        ax.set_ylabel("Number of Demes", fontsize=10)
        ax.set_title("Mean Test Accuracy", fontsize=11, fontweight='bold')
        for i in range(len(N_DEMES_ALL)):
            for j in range(len(MR_PCTS)):
                val = acc_grid[i, j]
                if not np.isnan(val):
                    sig = sig_acc[i][j]
                    marker = " ▲" if sig == "+" else (" ▼" if sig == "-" else "")
                    ax.text(j, i, f"{val:.4f}{marker}",
                            ha='center', va='center', fontsize=8.5,
                            fontweight='bold',
                            color='black')

        # ── BD panel (uses global colour scale for this dataset) ────────────
        ax = axes[1]
        masked_bd = np.ma.masked_invalid(bd_grid)
        # Use slightly padded global range so colour is consistent across freq
        bd_vmin = max(0, bd_global_min - 0.005)
        bd_vmax = bd_global_max + 0.005
        im2 = ax.imshow(masked_bd, cmap='YlOrRd', aspect='auto',
                        vmin=bd_vmin, vmax=bd_vmax)
        plt.colorbar(im2, ax=ax, shrink=0.85)
        ax.set_xticks(range(len(MR_PCTS)));   ax.set_xticklabels(mr_labels)
        ax.set_yticks(range(len(N_DEMES_ALL))); ax.set_yticklabels(nd_labels)
        ax.set_xlabel("Migration Rate", fontsize=10)
        ax.set_ylabel("Number of Demes", fontsize=10)
        ax.set_title("Mean Final Behavioural Diversity", fontsize=11, fontweight='bold')
        for i in range(len(N_DEMES_ALL)):
            for j in range(len(MR_PCTS)):
                val = bd_grid[i, j]
                if not np.isnan(val):
                    sig = sig_bd[i][j]
                    marker = " ▲" if sig == "+" else (" ▼" if sig == "-" else "")
                    ax.text(j, i, f"{val:.4f}{marker}",
                            ha='center', va='center', fontsize=8.5,
                            fontweight='bold',
                            color='white' if val > (bd_vmin + bd_vmax) * 0.55 else 'black')

        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"{dataset}_heatmap_freq{freq}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  plot_results_final3.py")
print("="*60)

print("\nPROFILE PLOTS")
print("─"*40)
plot_profile_bd_vs_mr()
plot_profile_bd_vs_ndemes()
plot_profile_bd_vs_freq()

print("\nGENERATIONAL PLOTS")
print("─"*40)
for ds in DATASETS:
    ds_path = os.path.join(RESULTS_ROOT, ds)
    if not os.path.exists(ds_path):
        print(f"  Skipping {ds} — no results folder")
        continue
    print(f"  {DS_LABELS[ds]}")
    plot_generational(ds)

print("\nHEATMAPS")
print("─"*40)
for ds in DATASETS:
    ds_path = os.path.join(RESULTS_ROOT, ds)
    if not os.path.exists(ds_path):
        continue
    print(f"  {DS_LABELS[ds]}")
    plot_heatmaps(ds)

print(f"\nAll figures saved to: {PLOTS_DIR}/")
