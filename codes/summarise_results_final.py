"""
summarise_results_final.py
Reads all CSVs from results_final/ and prints mean accuracy + mean final
behavioural diversity per dataset × configuration.

Usage (from inside the project root folder):
    python codes/summarise_results_final.py
"""
import os, glob
import numpy as np
import pandas as pd

RESULTS_ROOT = "results_final/GP"
DATASETS     = ["wine", "iris", "australian", "pima", "heartDisease"]

def load_metrics(folder):
    """
    Load the final-generation accuracy_test and behavioural_diversity
    values from every run CSV in folder.

    Parameters:
        folder -- directory containing per-run result CSVs (tab-separated).

    Returns:
        Tuple (accs, bds) of lists of floats, one entry per run that had
        a valid value for that metric.
    """
    accs, bds = [], []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            df  = pd.read_csv(f, sep='\t')
            acc = df["accuracy_test"].dropna()
            bd  = df["behavioural_diversity"].dropna()
            if len(acc) > 0: accs.append(float(acc.iloc[-1]))
            if len(bd)  > 0: bds.append(float(bd.iloc[-1]))
        except Exception as e:
            print(f"  Warning: {f}: {e}")
    return accs, bds

rows = []
print(f"\n{'Dataset':<14} {'Configuration':<30} {'N':>4} "
      f"{'MeanAcc':>9} {'StdAcc':>8} {'MeanBD':>8} {'StdBD':>7}")
print("─" * 88)

for ds in DATASETS:
    ds_path = os.path.join(RESULTS_ROOT, ds)
    if not os.path.exists(ds_path):
        continue
    subfolders = sorted(d for d in os.listdir(ds_path)
                        if os.path.isdir(os.path.join(ds_path, d)))
    for sub in subfolders:
        accs, bds = load_metrics(os.path.join(ds_path, sub))
        if not accs:
            continue
        m_acc = np.mean(accs); s_acc = np.std(accs)
        m_bd  = np.mean(bds)  if bds else np.nan
        s_bd  = np.std(bds)   if bds else np.nan
        print(f"{ds:<14} {sub:<30} {len(accs):>4} "
              f"{m_acc:>9.4f} {s_acc:>8.4f} {m_bd:>8.4f} {s_bd:>7.4f}")
        rows.append({"dataset": ds, "config": sub, "n_runs": len(accs),
                     "mean_acc": round(m_acc,4), "std_acc": round(s_acc,4),
                     "mean_bd":  round(m_bd, 4) if not np.isnan(m_bd) else np.nan,
                     "std_bd":   round(s_bd, 4) if not np.isnan(s_bd) else np.nan})

print()
out = os.path.join(RESULTS_ROOT, "summary_final.csv")
os.makedirs(RESULTS_ROOT, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved to {out}")
