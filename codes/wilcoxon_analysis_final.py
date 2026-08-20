"""
wilcoxon_analysis_final.py
==========================
Performs Wilcoxon signed-rank tests comparing every deme configuration
against the tournament baseline, per dataset.

Tests performed:
  - Accuracy  : deme config vs tournament (30 paired run values)
  - Behavioural Diversity : deme config vs tournament (30 paired run values)

Why Wilcoxon signed-rank and not a t-test?
  GP results are not guaranteed to be normally distributed. The Wilcoxon
  signed-rank test is a non-parametric alternative that makes no normality
  assumption. It is the standard choice in evolutionary computation research
  (see Demsar 2006, JMLR).

Why paired?
  Both the deme config and tournament use the same 30 random seeds
  (seeds 1..30). Each seed controls the train/test split AND the GP
  random initialisation. Pairing by seed removes the variance due to
  data split, giving a more sensitive test.

Significance level: alpha = 0.05 (standard in EC research).
Multiple comparisons: We report raw p-values. Bonferroni correction
  is noted where relevant but not applied by default, following common
  practice in EC comparative studies (Garcia et al. 2010).

Output:
  results_final/GP/wilcoxon_results.csv   — full per-config results
  results_final/GP/wilcoxon_summary.csv   — per-dataset win/tie/loss counts
  (also printed to terminal)

Usage (from inside the project root folder):
    python codes/wilcoxon_analysis_final.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_ROOT = "results_final/GP"
DATASETS     = ["wine", "iris", "australian", "pima", "heartDisease"]
ALPHA        = 0.05
N_RUNS       = 30


# ── Load per-run values from CSV files ───────────────────────────────────────
def load_run_values(folder, metric):
    """
    Load the per-run final value of `metric` from all CSVs in folder.
    Returns a list of 30 floats (one per seed), sorted by seed number.
    Returns None if folder missing or fewer than N_RUNS files found.
    """
    if not os.path.exists(folder):
        return None

    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(files) < N_RUNS:
        return None

    values = []
    for f in files[:N_RUNS]:
        try:
            df  = pd.read_csv(f, sep='\t')
            col = df[metric].dropna()
            if len(col) > 0:
                values.append(float(col.iloc[-1]))
        except Exception:
            pass

    return values if len(values) == N_RUNS else None


# ── Wilcoxon test helper ──────────────────────────────────────────────────────
def run_wilcoxon(vals_deme, vals_tourn):
    """
    Two-tailed Wilcoxon signed-rank test.
    Returns (statistic, p_value, direction, significance_marker).
    direction: '+' if deme > tournament, '-' if deme < tournament, '=' if equal
    """
    diff = np.array(vals_deme) - np.array(vals_tourn)

    # If all differences are zero, test cannot be performed
    if np.all(diff == 0):
        return np.nan, 1.0, '=', '='

    try:
        stat, p = wilcoxon(vals_deme, vals_tourn, alternative='two-sided',
                           zero_method='wilcox')
    except ValueError:
        # Raised when all differences are zero after zero_method handling
        return np.nan, 1.0, '=', '='

    mean_diff = np.mean(diff)
    if p < ALPHA:
        direction = '+' if mean_diff > 0 else '-'
        marker    = '+' if mean_diff > 0 else '-'
    else:
        direction = '+' if mean_diff > 0 else ('-' if mean_diff < 0 else '=')
        marker    = '='  # not significant

    return round(stat, 4), round(p, 4), direction, marker


# ── Main analysis ─────────────────────────────────────────────────────────────
rows_full    = []   # one row per dataset × config
rows_summary = []   # one row per dataset (win/tie/loss counts)

print("=" * 80)
print("WILCOXON SIGNED-RANK TESTS")
print(f"alpha = {ALPHA}  |  n = {N_RUNS} paired runs per config")
print("Marker: + = significantly better, - = significantly worse, = = no sig. diff.")
print("=" * 80)

for ds in DATASETS:
    print(f"\n{'─'*80}")
    print(f"DATASET: {ds.upper()}")
    print(f"{'─'*80}")

    # Load tournament baseline run values
    tourn_folder = os.path.join(RESULTS_ROOT, ds, "tournament")
    tourn_acc = load_run_values(tourn_folder, "accuracy_test")
    tourn_bd  = load_run_values(tourn_folder, "behavioural_diversity")

    if tourn_acc is None:
        print(f"  WARNING: tournament results missing for {ds} — skipping")
        continue

    tourn_acc_mean = np.mean(tourn_acc)
    tourn_bd_mean  = np.mean(tourn_bd) if tourn_bd else np.nan

    print(f"  Tournament baseline:  acc={tourn_acc_mean:.4f}  bd={tourn_bd_mean:.4f}")
    print()
    print(f"  {'Configuration':<30} {'MeanAcc':>8} {'W_acc':>8} {'p_acc':>8} "
          f"{'Sig_acc':>8} {'MeanBD':>8} {'W_bd':>8} {'p_bd':>8} {'Sig_bd':>8}")
    print(f"  {'-'*96}")

    # Counters for summary
    acc_wins = acc_ties = acc_losses = 0
    bd_wins  = bd_ties  = bd_losses  = 0

    # Find all deme config folders for this dataset
    ds_path = os.path.join(RESULTS_ROOT, ds)
    subfolders = sorted([
        d for d in os.listdir(ds_path)
        if os.path.isdir(os.path.join(ds_path, d))
        and d.startswith("demes_")
    ])

    for sub in subfolders:
        folder = os.path.join(ds_path, sub)

        deme_acc = load_run_values(folder, "accuracy_test")
        deme_bd  = load_run_values(folder, "behavioural_diversity")

        if deme_acc is None:
            print(f"  {sub:<30} INCOMPLETE — skipping")
            continue

        # Accuracy test
        W_acc, p_acc, dir_acc, sig_acc = run_wilcoxon(deme_acc, tourn_acc)
        mean_acc = np.mean(deme_acc)

        # BD test
        if deme_bd and tourn_bd:
            W_bd, p_bd, dir_bd, sig_bd = run_wilcoxon(deme_bd, tourn_bd)
            mean_bd = np.mean(deme_bd)
        else:
            W_bd = p_bd = mean_bd = np.nan
            sig_bd = '?'

        # Update summary counters
        if   sig_acc == '+': acc_wins   += 1
        elif sig_acc == '-': acc_losses += 1
        else:                acc_ties   += 1

        if   sig_bd == '+':  bd_wins   += 1
        elif sig_bd == '-':  bd_losses += 1
        else:                bd_ties   += 1

        print(f"  {sub:<30} {mean_acc:>8.4f} {W_acc:>8} {p_acc:>8.4f} "
              f"{sig_acc:>8} {mean_bd:>8.4f} {W_bd:>8} {p_bd:>8.4f} {sig_bd:>8}")

        rows_full.append({
            'dataset':       ds,
            'config':        sub,
            'mean_acc_deme': round(mean_acc, 4),
            'mean_acc_tourn':round(tourn_acc_mean, 4),
            'acc_diff':      round(mean_acc - tourn_acc_mean, 4),
            'W_acc':         W_acc,
            'p_acc':         p_acc,
            'sig_acc':       sig_acc,
            'mean_bd_deme':  round(mean_bd, 4) if not np.isnan(mean_bd) else np.nan,
            'mean_bd_tourn': round(tourn_bd_mean, 4),
            'bd_diff':       round(mean_bd - tourn_bd_mean, 4) if not np.isnan(mean_bd) else np.nan,
            'W_bd':          W_bd,
            'p_bd':          p_bd,
            'sig_bd':        sig_bd,
        })

    # Dataset summary
    n_configs = acc_wins + acc_ties + acc_losses
    print(f"\n  ACCURACY  — wins(+): {acc_wins}  ties(=): {acc_ties}  "
          f"losses(-): {acc_losses}  /  {n_configs} configs")
    print(f"  BD        — wins(+): {bd_wins}  ties(=): {bd_ties}  "
          f"losses(-): {bd_losses}  /  {n_configs} configs")

    rows_summary.append({
        'dataset':       ds,
        'n_configs':     n_configs,
        'tourn_acc':     round(tourn_acc_mean, 4),
        'tourn_bd':      round(tourn_bd_mean, 4),
        'acc_wins':      acc_wins,
        'acc_ties':      acc_ties,
        'acc_losses':    acc_losses,
        'bd_wins':       bd_wins,
        'bd_ties':       bd_ties,
        'bd_losses':     bd_losses,
    })

# ── Save outputs ──────────────────────────────────────────────────────────────
os.makedirs(RESULTS_ROOT, exist_ok=True)

out_full = os.path.join(RESULTS_ROOT, "wilcoxon_results.csv")
pd.DataFrame(rows_full).to_csv(out_full, index=False)

out_summary = os.path.join(RESULTS_ROOT, "wilcoxon_summary.csv")
pd.DataFrame(rows_summary).to_csv(out_summary, index=False)

# ── Overall summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("OVERALL SUMMARY — Accuracy (Wilcoxon, alpha=0.05)")
print("=" * 80)
print(f"{'Dataset':<15} {'N configs':>10} {'+ wins':>8} {'= ties':>8} {'- losses':>10}")
print("-" * 55)
for r in rows_summary:
    print(f"  {r['dataset']:<13} {r['n_configs']:>10} {r['acc_wins']:>8} "
          f"{r['acc_ties']:>8} {r['acc_losses']:>10}")

print("\n" + "=" * 80)
print("OVERALL SUMMARY — Behavioural Diversity (Wilcoxon, alpha=0.05)")
print("=" * 80)
print(f"{'Dataset':<15} {'N configs':>10} {'+ wins':>8} {'= ties':>8} {'- losses':>10}")
print("-" * 55)
for r in rows_summary:
    print(f"  {r['dataset']:<13} {r['n_configs']:>10} {r['bd_wins']:>8} "
          f"{r['bd_ties']:>8} {r['bd_losses']:>10}")

print(f"\nFull results saved to : {out_full}")
print(f"Summary saved to      : {out_summary}")
print("\nNote: Use wilcoxon_results.csv for the dissertation Results chapter.")
print("      Use wilcoxon_summary.csv for the summary table in Results/Discussion.")
