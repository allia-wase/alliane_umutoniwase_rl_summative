"""Run every hyperparameter configuration of every algorithm, N in parallel.

    uv run python -m training.run_sweeps                 # all algorithms
    uv run python -m training.run_sweeps --algos dqn ppo --workers 4

Finished runs (a *_summary.json exists) are skipped, so it can be resumed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.common import LOG_DIR
from training.dqn_training import SWEEP as DQN_SWEEP
from training.pg_training import SWEEPS as PG_SWEEPS

ROOT = Path(__file__).resolve().parents[1]


def jobs(algos):
    for algo in algos:
        names = DQN_SWEEP if algo == "dqn" else PG_SWEEPS[algo]
        for name in names:
            if (LOG_DIR / algo / f"{name}_summary.json").exists():
                continue
            if algo == "dqn":
                cmd = [sys.executable, "-m", "training.dqn_training", "--run", name]
            else:
                cmd = [sys.executable, "-m", "training.pg_training", "--algo", algo, "--run", name]
            yield algo, name, cmd


def run(job):
    algo, name, cmd = job
    out = LOG_DIR / algo / f"{name}.stdout.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    tail = out.read_text(errors="ignore").strip().splitlines()[-1:]
    print(f"{'OK ' if rc == 0 else 'FAIL'} {algo}/{name}: {tail[0] if tail else ''}", flush=True)
    return rc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=["dqn", "reinforce", "a2c", "ppo"])
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    todo = list(jobs(a.algos))
    print(f"{len(todo)} runs to do with {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(run, todo))
    print(f"done: {codes.count(0)} ok, {len(codes) - codes.count(0)} failed")
