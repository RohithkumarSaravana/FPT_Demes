# Fuzzy Genetic Programming with Island-Model (Demes) Selection

This repository contains the code used for the dissertation experiments comparing a
standard tournament-selection Genetic Programming (GP) classifier against an
island-model ("demes") variant with migration, across five benchmark datasets.

Each individual is a fuzzy-rule-based GP tree, evolved with
[DEAP](https://deap.readthedocs.io/) and evaluated on classification accuracy and
behavioural diversity (the fraction of unique prediction vectors in the population).

**Repository:** https://github.com/RohithkumarSaravana/FPT_Demes

## Repository structure

```
finalgp/
├── codes/                          # All source code (see below)
├── datasets/                       # Raw benchmark datasets (downloaded, see step 1)
├── results_final/GP/               # Experiment output (created when experiments run)
│   ├── <dataset>/tournament/           # Baseline runs
│   ├── <dataset>/demes_<n>_mr<pct>_freq<f>/   # Island-model runs, one folder per config
│   ├── summary_final.csv
│   ├── wilcoxon_results.csv / wilcoxon_summary.csv
│   ├── table_A_accuracy.csv / table_B_bd.csv / table_C_accuracy_by_param.csv / table_D_bd_by_param.csv
│   └── plots/                          # Generated figures
└── README.md
```

## Datasets

Five standard UCI/benchmark classification datasets:

| Dataset | Classes | File |
|---|---|---|
| Wine | 3 | `wine.data` |
| Iris | 3 | `iris.data` |
| Australian Credit Approval | 2 | `australian.dat` |
| Pima Indians Diabetes | 2 | `pima.data` |
| Heart Disease (Cleveland) | 2 | `processed.cleveland.data` |

## 1. Setup

**Requirements:** Python 3.10+ (the code was developed and tested on Python 3.10).

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn scipy matplotlib deap scikit-fuzzy
```

Download the datasets into `datasets/` (run once, from the project root):

```bash
python codes/download_datasets_final.py
```

This fetches `australian.dat`, `processed.cleveland.data`, and `pima.data` from
their public mirrors, and prints a verification summary. `iris.data` and
`wine.data` are already included in `datasets/`.

## 2. Core algorithm code

These modules are imported by the experiment scripts and are not run directly:

| File | Purpose |
|---|---|
| `codes/functions.py` | Fuzzy-set operators (WA, OWA, dilators, concentrators, AND/OR/complement) and small array-manipulation helpers used by the GP primitive set and by lexicase selection. |
| `codes/fuzzify.py` | Builds the domain matrix for each dataset feature and fuzzifies a DataFrame into fuzzy-set membership values used as GP terminals. |
| `codes/algorithms_gp_final.py` | The evolutionary loop: `eaSimple` (standard single-population GP) and `eaSimple_demes` (island-model GP with circular migration). |
| `codes/selection.py` | Lexicase selection and its variants (epsilon-lexicase, batch-lexicase, dynamic-epsilon, node-count tie-breaking, etc.). Not all variants are used by the final experiments — kept for completeness/reference. |

## 3. Running experiments

All commands below are run **from the project root** (`finalgp/`).

### Single experiment

Tournament baseline, one dataset, 30 runs:

```bash
python codes/paper_complete_script_final.py wine 1 30
```

Arguments: `<dataset> <run_start> <n_runs>`. Datasets: `wine`, `iris`, `australian`, `pima`, `heartDisease`.

Island-model (demes), one configuration:

```bash
python codes/demes_experiment_final.py wine 10 5 1 1 30
```

Arguments: `<dataset> <n_demes> <migration_rate_pct> <migration_frequency> [run_start] [n_runs]`.

- `n_demes`: 5, 10, or 15
- `migration_rate_pct`: 2, 5, 10, or 15 (percent of a deme that migrates)
- `migration_frequency`: 1, 5, 10, or 25 (migration fires every N generations)

Results are written as one CSV per run to `results_final/GP/<dataset>/<config>/`.

### Full experiment grid (recommended)

`run_experiments_final.py` runs every combination (baseline + all demes configs ×
all datasets × 30 runs) in parallel subprocesses:

```bash
python codes/run_experiments_final.py                    # everything
python codes/run_experiments_final.py --only baseline     # tournament baseline only
python codes/run_experiments_final.py --only demes        # all demes configs
python codes/run_experiments_final.py --only freq1        # demes configs with freq=1 only
python codes/run_experiments_final.py --dataset wine       # restrict to one dataset
python codes/run_experiments_final.py --max-workers 4      # limit parallel processes
python codes/run_experiments_final.py --resume             # skip configs already complete
```

This is a long run (the full grid is 5 datasets × (1 baseline + 3×4×4 demes
configs) × 30 seeds). Use `--max-workers` to match your machine's core count, and
`--resume` to continue an interrupted run. A progress log is written to
`results_final/GP/run_log.txt`.

## 4. Analysis and output generation

Run these **after** the experiments above have produced results, in this order:

```bash
# 1. Aggregate mean accuracy / behavioural diversity per dataset × configuration
python codes/summarise_results_final.py

# 2. Wilcoxon signed-rank tests: each demes config vs. the tournament baseline
python codes/wilcoxon_analysis_final.py

# 3. Dissertation-ready result tables (CSV), built from the two outputs above
python codes/generate_tables_final.py

# 4. Figures: generational curves, profile plots, and accuracy/BD heatmaps
python codes/plot_results_final3.py
```

Outputs:

- `results_final/GP/summary_final.csv` — mean accuracy and final behavioural
  diversity per dataset/configuration.
- `results_final/GP/wilcoxon_results.csv`, `wilcoxon_summary.csv` — significance
  test results and per-dataset win/tie/loss counts.
- `results_final/GP/table_A_accuracy.csv`, `table_B_bd.csv`,
  `table_C_accuracy_by_param.csv`, `table_D_bd_by_param.csv` — summary tables.
- `results_final/GP/plots/*.png` — generational curves, BD profile plots, and
  accuracy/BD heatmaps per dataset.

## Notes on the experimental design

- Every run uses a fixed seed (1..30) that controls both the train/test split and
  GP's random initialisation, so tournament and demes runs are paired for the
  Wilcoxon test.
- The Wilcoxon signed-rank test (not a t-test) is used because GP results are not
  guaranteed to be normally distributed; this is standard practice in
  evolutionary computation research.
- Behavioural diversity is defined as the fraction of unique prediction vectors
  in the population.
