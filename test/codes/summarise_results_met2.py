"""
summarise_results_met2.py
Reads all CSV files from results_met2/ and produces a summary table with:
  - mean accuracy_test over 30 runs
  - mean final behavioural_diversity over 30 runs

Usage (run from inside the met2/ folder):
    python codes/summarise_results_met2.py
"""

import os, glob
import numpy as np
import pandas as pd

RESULTS_ROOT = "results_met2/GP"
DATASETS     = ["wine", "iris", "australian", "pima", "heart"]

def load_metrics(folder_path):
    """
    Returns (accuracies, final_bd) lists from all CSVs in folder_path.
    accuracy_test is the non-NaN value in the last generation.
    behavioural_diversity final value is the last row's value.
    """
    accuracies = []
    final_bds  = []
    for f in sorted(glob.glob(os.path.join(folder_path, "*.csv"))):
        try:
            df = pd.read_csv(f, sep='\t')
            acc_vals = df["accuracy_test"].dropna()
            if len(acc_vals) > 0:
                accuracies.append(float(acc_vals.iloc[-1]))
            bd_vals = df["behavioural_diversity"].dropna()
            if len(bd_vals) > 0:
                final_bds.append(float(bd_vals.iloc[-1]))
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    return accuracies, final_bds

rows = []
header = f"\n{'Dataset':<12} {'Configuration':<22} {'Runs':>5} " \
         f"{'Mean Acc':>10} {'Std Acc':>8} {'Mean BD':>9} {'Std BD':>8}"
print(header)
print("-" * 80)

for dataset in DATASETS:
    dataset_path = os.path.join(RESULTS_ROOT, dataset)
    if not os.path.exists(dataset_path):
        continue

    # Find all subfolders (tournament + all demes configs)
    subfolders = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    for subfolder in subfolders:
        folder = os.path.join(dataset_path, subfolder)
        accs, bds = load_metrics(folder)
        if not accs:
            continue

        mean_acc = np.mean(accs);  std_acc = np.std(accs)
        mean_bd  = np.mean(bds)  if bds else float('nan')
        std_bd   = np.std(bds)   if bds else float('nan')
        n_runs   = len(accs)

        label = subfolder.replace("_", " ").replace("demes", "Demes").replace("mr", "mr=")
        if label == "tournament":
            label = "Tournament (baseline)"

        print(f"{dataset:<12} {label:<22} {n_runs:>5} "
              f"{mean_acc:>10.4f} {std_acc:>8.4f} {mean_bd:>9.4f} {std_bd:>8.4f}")

        rows.append({
            "dataset":   dataset,
            "config":    subfolder,
            "n_runs":    n_runs,
            "mean_acc":  round(mean_acc, 4),
            "std_acc":   round(std_acc,  4),
            "mean_bd":   round(mean_bd,  4) if not np.isnan(mean_bd) else float('nan'),
            "std_bd":    round(std_bd,   4) if not np.isnan(std_bd)  else float('nan'),
        })

print()
out_dir = RESULTS_ROOT
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "summary_met2.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"Summary saved to {out_path}")
