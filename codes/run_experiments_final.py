"""
run_experiments_final.py
=======================
Runs all experiments in parallel using Python's multiprocessing.
Each experiment is a subprocess — they run independently and write to
their own result folders. No conflicts.

Usage (from inside the project root folder):
    python codes/run_experiments_final.py                  # all experiments
    python codes/run_experiments_final.py --max-workers 4  # limit parallelism
    python codes/run_experiments_final.py --only baseline  # only baseline
    python codes/run_experiments_final.py --only demes     # only demes
    python codes/run_experiments_final.py --only freq1     # only freq=1 demes
    python codes/run_experiments_final.py --dataset wine   # one dataset only
    python codes/run_experiments_final.py --resume         # skip already-done runs

Controls:
    MAX_WORKERS   — how many experiments run simultaneously
                    default = number of CPU cores - 1 (leaves one core free)
                    Mac M-series: 8-10 is comfortable
                    Adjust with --max-workers N
"""

import subprocess
import sys
import os
import argparse
import time
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

# ── Experiment grid ───────────────────────────────────────────────────────────
DATASETS    = ["wine", "iris", "australian", "pima", "heartDisease"]
N_DEMES_ALL = [5, 10, 15]
MR_PCTS     = [2, 5, 10, 15]
FREQS       = [1, 5, 10, 25]
N_RUNS      = 30
RUN_START   = 1

RESULTS_ROOT = "results_final/GP"

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run all experiments in parallel")
parser.add_argument("--max-workers", type=int, default=None,
                    help="Max parallel experiments (default: cpu_count - 1)")
parser.add_argument("--only", type=str, default="all",
                    choices=["all", "baseline", "demes", "freq1", "freq5",
                             "freq10", "freq25"],
                    help="Which experiments to run")
parser.add_argument("--dataset", type=str, default=None,
                    choices=DATASETS,
                    help="Run only this dataset")
parser.add_argument("--resume", action="store_true",
                    help="Skip configs that already have 30 result CSVs")
args = parser.parse_args()

# Default workers: leave 1 core free for the OS
import multiprocessing
MAX_WORKERS = args.max_workers or max(1, multiprocessing.cpu_count() - 1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_complete(folder, n_runs=N_RUNS):
    """Return True if the folder already has n_runs CSV files."""
    if not os.path.exists(folder):
        return False
    csvs = glob.glob(os.path.join(folder, "*.csv"))
    return len(csvs) >= n_runs

def run_cmd(cmd_info):
    """
    Runs a single experiment as a subprocess.
    cmd_info = (label, cmd_list)
    Returns (label, returncode, duration_seconds, stdout_tail)
    """
    label, cmd = cmd_info
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200   # 2 hour max per experiment
        )
        duration = time.time() - t0
        # Return last 3 lines of stdout as summary
        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        tail  = '\n'.join(lines[-3:]) if lines else "(no output)"
        return label, result.returncode, duration, tail
    except subprocess.TimeoutExpired:
        return label, -1, time.time() - t0, "TIMEOUT"
    except Exception as e:
        return label, -2, time.time() - t0, str(e)

# ── Build experiment list ─────────────────────────────────────────────────────
def build_experiments():
    """
    Build the list of (label, cmd) experiments to run, based on the
    parsed CLI args: which dataset(s), which subset (baseline / demes /
    a specific migration frequency / all), and whether to skip configs
    already complete (--resume). Baseline runs invoke
    paper_complete_script_final.py; island-model runs invoke
    demes_experiment_final.py with the (n_demes, mr, freq) grid.

    Returns:
        List of (label, cmd) tuples, where cmd is the subprocess argv
        list for run_cmd.
    """
    experiments = []
    datasets = [args.dataset] if args.dataset else DATASETS

    # ── Baseline ──────────────────────────────────────────────────────────────
    if args.only in ("all", "baseline"):
        for ds in datasets:
            folder = os.path.join(RESULTS_ROOT, ds, "tournament")
            if args.resume and is_complete(folder):
                print(f"  [SKIP] {ds} tournament — already complete")
                continue
            label = f"baseline | {ds}"
            cmd   = [sys.executable, "codes/paper_complete_script_final.py",
                     ds, str(RUN_START), str(N_RUNS)]
            experiments.append((label, cmd))

    # ── Demes ─────────────────────────────────────────────────────────────────
    freq_filter = None
    if args.only == "demes":
        freq_filter = FREQS
    elif args.only == "freq1":
        freq_filter = [1]
    elif args.only == "freq5":
        freq_filter = [5]
    elif args.only == "freq10":
        freq_filter = [10]
    elif args.only == "freq25":
        freq_filter = [25]
    elif args.only == "all":
        freq_filter = FREQS

    if freq_filter is not None:
        for freq in freq_filter:
            for nd in N_DEMES_ALL:
                for mr in MR_PCTS:
                    for ds in datasets:
                        config = f"demes_{nd}_mr{mr}_freq{freq}"
                        folder = os.path.join(RESULTS_ROOT, ds, config)
                        if args.resume and is_complete(folder):
                            print(f"  [SKIP] {ds} {config} — already complete")
                            continue
                        label = f"{ds} | n={nd} mr={mr}% freq={freq}"
                        cmd   = [sys.executable, "codes/demes_experiment_final.py",
                                 ds, str(nd), str(mr), str(freq),
                                 str(RUN_START), str(N_RUNS)]
                        experiments.append((label, cmd))

    return experiments

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    """
    Entry point: builds the experiment list, runs it across a process
    pool (up to MAX_WORKERS concurrent subprocesses), logs progress and
    failures to results_final/GP/run_log.txt, and — if every experiment
    succeeded and the full grid was run — automatically runs
    summarise_results_final.py afterwards.
    """
    print("=" * 70)
    print("  Experiment Runner")
    print("=" * 70)
    print(f"  Mode       : {args.only}")
    print(f"  Dataset    : {args.dataset or 'all'}")
    print(f"  Max workers: {MAX_WORKERS}")
    print(f"  Resume     : {args.resume}")
    print()

    experiments = build_experiments()
    total = len(experiments)

    if total == 0:
        print("No experiments to run (all may already be complete).")
        return

    print(f"Experiments to run: {total}")
    print(f"Running {MAX_WORKERS} in parallel\n")
    print("-" * 70)

    # Estimate time
    # Each 30-run experiment takes ~5-10 min on M-series Mac
    est_serial   = total * 7   # minutes, rough estimate per experiment
    est_parallel = est_serial / MAX_WORKERS
    print(f"  Rough time estimate: {int(est_parallel)} – {int(est_parallel*1.5)} minutes "
          f"(running {MAX_WORKERS} in parallel)")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Estimated finish: "
          f"{(datetime.now() + timedelta(minutes=est_parallel)).strftime('%H:%M:%S')}")
    print("-" * 70)
    print()

    completed = 0
    failed    = []
    t_start   = time.time()

    # Write a log file
    log_path = os.path.join(RESULTS_ROOT, "run_log.txt")
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    with open(log_path, "w") as log:
        log.write(f"Run started: {datetime.now()}\n")
        log.write(f"Total experiments: {total}\n")
        log.write(f"Max workers: {MAX_WORKERS}\n\n")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(run_cmd, exp): exp for exp in experiments}

            for future in as_completed(futures):
                label, returncode, duration, tail = future.result()
                completed += 1
                elapsed = time.time() - t_start
                avg_per_exp = elapsed / completed
                remaining   = (total - completed) * avg_per_exp

                status = "✓" if returncode == 0 else "✗"
                mins   = int(duration // 60)
                secs   = int(duration % 60)

                print(f"[{completed:>4}/{total}] {status}  {label:<40}  "
                      f"{mins}m{secs:02d}s  "
                      f"ETA: {int(remaining//60)}m{int(remaining%60):02d}s")

                if returncode != 0:
                    failed.append(label)
                    print(f"         ERROR (code {returncode}):")
                    for line in tail.split('\n'):
                        print(f"           {line}")

                log.write(f"[{completed}/{total}] {status} {label} "
                          f"({mins}m{secs:02d}s, code={returncode})\n")
                if returncode != 0:
                    log.write(f"  OUTPUT: {tail}\n")
                log.flush()

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print(f"  Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Total time: {int(total_time//60)}m{int(total_time%60):02d}s")
    print(f"  Completed: {completed - len(failed)}/{total}")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
    else:
        print("  All experiments completed successfully!")
    print(f"  Log saved to: {log_path}")
    print("=" * 70)

    # Auto-run summary after all experiments
    if not failed and args.only in ("all",):
        print("\nRunning summary...")
        subprocess.run([sys.executable, "codes/summarise_results_final.py"])

if __name__ == "__main__":
    main()
