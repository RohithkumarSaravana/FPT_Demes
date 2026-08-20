"""
generate_tables_final.py
========================
Reads wilcoxon_results.csv and summary_final.csv and produces
dissertation-ready result tables as CSV files.

Tables produced:

  Table A — Main accuracy results table
    Rows: datasets
    Cols: tournament | demes configs (grouped by n_demes)
    Cells: mean_acc (std_acc) marker
    Markers: + significantly better, - significantly worse, = no difference

  Table B — Behavioural diversity results table
    Same layout as Table A but for BD

  Table C — Summary: mean accuracy per parameter value
    Shows mean accuracy averaged across all configs that share each
    parameter value (n_demes, mr, freq), for each dataset.
    This directly answers the research question: which parameters matter?

  Table D — Summary: mean BD per parameter value
    Same as Table C but for BD.

Usage (from inside the project root folder):
    python codes/generate_tables_final.py
    (must run wilcoxon_analysis_final.py first)
"""

import os
import numpy as np
import pandas as pd

RESULTS_ROOT = "results_final/GP"
DATASETS     = ["wine", "iris", "australian", "pima", "heartDisease"]

# ── Load data ─────────────────────────────────────────────────────────────────
wil_path = os.path.join(RESULTS_ROOT, "wilcoxon_results.csv")
sum_path = os.path.join(RESULTS_ROOT, "summary_final.csv")

if not os.path.exists(wil_path):
    print(f"ERROR: {wil_path} not found.")
    print("Run wilcoxon_analysis_final.py first.")
    exit(1)

wil = pd.read_csv(wil_path)
summ = pd.read_csv(sum_path)

# Parse config name into components
def parse_config(cfg):
    """
    Parse a config label into its (n_demes, mr, freq) components.

    Parameters:
        cfg -- config string, either 'tournament' (baseline) or
               'demes_{n}_mr{mr}_freq{freq}'.

    Returns:
        Tuple (n_demes, mr, freq) as ints, or (None, None, None) for the
        tournament baseline.
    """
    if cfg == 'tournament':
        return None, None, None
    parts = cfg.split('_')
    # demes_{n}_mr{mr}_freq{f}
    nd   = int(parts[1])
    mr   = int(parts[2].replace('mr',''))
    freq = int(parts[3].replace('freq',''))
    return nd, mr, freq

wil[['n_demes','mr','freq']] = wil['config'].apply(
    lambda x: pd.Series(parse_config(x)))
summ[['n_demes','mr','freq']] = summ['config'].apply(
    lambda x: pd.Series(parse_config(x)))

os.makedirs(RESULTS_ROOT, exist_ok=True)

# ── Table A: Accuracy results — mean(std) marker ──────────────────────────────
print("Generating Table A: Accuracy results...")

# Build rows: one per dataset, cols = tournament + all 48 deme configs
tourn_rows = summ[summ['config']=='tournament'][
    ['dataset','mean_acc','std_acc']].copy()
tourn_rows.columns = ['dataset','tourn_acc','tourn_std']

# Group deme results by (n_demes, mr, freq) for compact display
# We show mean accuracy and the Wilcoxon marker
tableA_rows = []
for ds in DATASETS:
    tourn_acc = tourn_rows[tourn_rows['dataset']==ds]['tourn_acc'].values[0]
    tourn_std = tourn_rows[tourn_rows['dataset']==ds]['tourn_std'].values[0]
    row = {'dataset': ds,
           'tournament': f"{tourn_acc:.2f}({tourn_std:.2f})"}

    sub = wil[wil['dataset']==ds].copy()
    for _, r in sub.iterrows():
        key = f"n{int(r['n_demes'])}_mr{int(r['mr'])}_f{int(r['freq'])}"
        cell = f"{r['mean_acc_deme']:.2f}({r['acc_diff']:+.3f}){r['sig_acc']}"
        row[key] = cell
    tableA_rows.append(row)

tableA = pd.DataFrame(tableA_rows)
out_a = os.path.join(RESULTS_ROOT, "table_A_accuracy.csv")
tableA.to_csv(out_a, index=False)
print(f"  Saved: {out_a}")

# ── Table B: BD results ────────────────────────────────────────────────────────
print("Generating Table B: BD results...")

tourn_bd_rows = summ[summ['config']=='tournament'][
    ['dataset','mean_bd','std_bd']].copy()
tourn_bd_rows.columns = ['dataset','tourn_bd','tourn_bd_std']

tableB_rows = []
for ds in DATASETS:
    tourn_bd  = tourn_bd_rows[tourn_bd_rows['dataset']==ds]['tourn_bd'].values[0]
    tourn_std = tourn_bd_rows[tourn_bd_rows['dataset']==ds]['tourn_bd_std'].values[0]
    row = {'dataset': ds,
           'tournament': f"{tourn_bd:.3f}({tourn_std:.3f})"}
    sub = wil[wil['dataset']==ds].copy()
    for _, r in sub.iterrows():
        key = f"n{int(r['n_demes'])}_mr{int(r['mr'])}_f{int(r['freq'])}"
        bd_diff = r['bd_diff'] if not pd.isna(r['bd_diff']) else 0
        cell = f"{r['mean_bd_deme']:.3f}({bd_diff:+.3f}){r['sig_bd']}"
        row[key] = cell
    tableB_rows.append(row)

tableB = pd.DataFrame(tableB_rows)
out_b = os.path.join(RESULTS_ROOT, "table_B_bd.csv")
tableB.to_csv(out_b, index=False)
print(f"  Saved: {out_b}")

# ── Table C: Mean accuracy per parameter value ────────────────────────────────
print("Generating Table C: Mean accuracy per parameter...")

deme_summ = summ[summ['config']!='tournament'].copy()
deme_summ[['n_demes','mr','freq']] = deme_summ['config'].apply(
    lambda x: pd.Series(parse_config(x)))

tableC_rows = []

for ds in DATASETS:
    sub = deme_summ[deme_summ['dataset']==ds]
    tourn_acc = summ[(summ['dataset']==ds) &
                     (summ['config']=='tournament')]['mean_acc'].values[0]
    row = {'dataset': ds, 'tournament': round(tourn_acc, 4)}

    # Average accuracy by n_demes
    for nd in [5, 10, 15]:
        avg = sub[sub['n_demes']==nd]['mean_acc'].mean()
        row[f'n_demes={nd}'] = round(avg, 4)

    # Average accuracy by migration rate
    for mr in [2, 5, 10, 15]:
        avg = sub[sub['mr']==mr]['mean_acc'].mean()
        row[f'mr={mr}%'] = round(avg, 4)

    # Average accuracy by frequency
    for freq in [1, 5, 10, 25]:
        avg = sub[sub['freq']==freq]['mean_acc'].mean()
        row[f'freq={freq}'] = round(avg, 4)

    tableC_rows.append(row)

tableC = pd.DataFrame(tableC_rows)
out_c = os.path.join(RESULTS_ROOT, "table_C_accuracy_by_param.csv")
tableC.to_csv(out_c, index=False)
print(f"  Saved: {out_c}")

# ── Table D: Mean BD per parameter value ─────────────────────────────────────
print("Generating Table D: Mean BD per parameter...")

tableD_rows = []
for ds in DATASETS:
    sub = deme_summ[deme_summ['dataset']==ds]
    tourn_bd = summ[(summ['dataset']==ds) &
                    (summ['config']=='tournament')]['mean_bd'].values[0]
    row = {'dataset': ds, 'tournament': round(tourn_bd, 4)}

    for nd in [5, 10, 15]:
        avg = sub[sub['n_demes']==nd]['mean_bd'].mean()
        row[f'n_demes={nd}'] = round(avg, 4)

    for mr in [2, 5, 10, 15]:
        avg = sub[sub['mr']==mr]['mean_bd'].mean()
        row[f'mr={mr}%'] = round(avg, 4)

    for freq in [1, 5, 10, 25]:
        avg = sub[sub['freq']==freq]['mean_bd'].mean()
        row[f'freq={freq}'] = round(avg, 4)

    tableD_rows.append(row)

tableD = pd.DataFrame(tableD_rows)
out_d = os.path.join(RESULTS_ROOT, "table_D_bd_by_param.csv")
tableD.to_csv(out_d, index=False)
print(f"  Saved: {out_d}")

# ── Print Table C and D to terminal ──────────────────────────────────────────
print("\n" + "="*80)
print("TABLE C — Mean Accuracy by Parameter Value")
print("="*80)
print(tableC.to_string(index=False))

print("\n" + "="*80)
print("TABLE D — Mean Behavioural Diversity by Parameter Value")
print("="*80)
print(tableD.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS FROM TABLE C (accuracy):")
print("="*80)
for ds in DATASETS:
    row = tableC[tableC['dataset']==ds].iloc[0]
    tourn = row['tournament']
    nd_vals  = {nd: row[f'n_demes={nd}']  for nd in [5,10,15]}
    mr_vals  = {mr: row[f'mr={mr}%']      for mr in [2,5,10,15]}
    fr_vals  = {f:  row[f'freq={f}']      for f  in [1,5,10,25]}
    best_nd  = max(nd_vals, key=nd_vals.get)
    best_mr  = max(mr_vals, key=mr_vals.get)
    best_fr  = max(fr_vals, key=fr_vals.get)
    print(f"  {ds}: tourn={tourn:.4f}  "
          f"best n_demes={best_nd}({nd_vals[best_nd]:.4f})  "
          f"best mr={best_mr}%({mr_vals[best_mr]:.4f})  "
          f"best freq={best_fr}({fr_vals[best_fr]:.4f})")

print("\n" + "="*80)
print("KEY FINDINGS FROM TABLE D (BD):")
print("="*80)
for ds in DATASETS:
    row = tableD[tableD['dataset']==ds].iloc[0]
    tourn = row['tournament']
    nd_vals  = {nd: row[f'n_demes={nd}']  for nd in [5,10,15]}
    mr_vals  = {mr: row[f'mr={mr}%']      for mr in [2,5,10,15]}
    fr_vals  = {f:  row[f'freq={f}']      for f  in [1,5,10,25]}
    best_nd  = max(nd_vals, key=nd_vals.get)
    best_mr  = max(mr_vals, key=mr_vals.get)
    best_fr  = max(fr_vals, key=fr_vals.get)
    print(f"  {ds}: tourn={tourn:.4f}  "
          f"best n_demes={best_nd}({nd_vals[best_nd]:.4f})  "
          f"best mr={best_mr}%({mr_vals[best_mr]:.4f})  "
          f"best freq={best_fr}({fr_vals[best_fr]:.4f})")

print(f"\nAll tables saved to {RESULTS_ROOT}/")
