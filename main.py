"""SafePath: run the best trained agent in the OpenGL simulation.

    uv run main.py                          # best safe model, 5 episodes, GUI
    uv run main.py --model models/pg/ppo_baseline.pt --episodes 3
    uv run main.py --random                 # untrained random agent for comparison
    uv run main.py --no-render              # terminal only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import ACTIONS, SafePathEnv
from training.common import load_model

ROOT = Path(__file__).resolve().parent


def best_model_path() -> Path:
    """Pick the highest-return run that hands >= 90% of at-risk survivors to a human.

    The safety rule matters: under the original reward some top-scoring agents learned to skip
    the safety screen. Only sweep runs (reward v1) are compared, so returns are on one scale.
    """
    best, best_score = None, -1e9
    for summ in (ROOT / "logs").glob("*/*_summary.json"):
        s = json.loads(summ.read_text())
        if s["risk_handled_rate"] < 0.9 or s["config"].get("reward_version", 1) != 1:
            continue
        kind = "dqn" if s["algo"] == "dqn" else "pg"
        pt = ROOT / "models" / kind / f"{s['algo']}_{s['run']}.pt"
        if pt.exists() and s["eval_mean_return"] > best_score:
            best, best_score = pt, s["eval_mean_return"]
    if best is None:
        raise SystemExit("No trained models found. Train first, e.g. "
                         "`uv run python -m training.pg_training --algo ppo`.")
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--delay", type=float, default=0.9, help="seconds between steps in the GUI")
    p.add_argument("--random", action="store_true")
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args()

    if a.random:
        net, meta = None, {"algo": "random", "run": "-"}
    else:
        path = Path(a.model) if a.model else best_model_path()
        net, meta = load_model(path)
        print(f"Loaded {meta['algo'].upper()} model '{meta['run']}' from {path}")

    env = SafePathEnv(render_mode=None if a.no_render else "human")
    rng = np.random.default_rng(a.seed)
    totals = []
    for ep in range(a.episodes):
        obs, info = env.reset(seed=a.seed + ep)
        print(f"\n=== Episode {ep + 1}: hidden profile -> risk={info['risk']} "
              f"shared_device={info['shared_device']} wants={info['preference']} "
              f"distress={info['distress']:.2f} trust={info['trust']:.2f}")
        done, R = False, 0.0
        while not done:
            if net is None:
                act = int(rng.integers(len(ACTIONS)))
            else:
                with torch.no_grad():
                    act = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))
            obs, r, term, trunc, info = env.step(act)
            R += r
            done = term or trunc
            print(f"  t={env.t:02d} {ACTIONS[act]:<16} r={r:+6.2f}  trust={info['trust']:.2f} "
                  f"distress={info['distress']:.2f} readiness={info['readiness']:.2f} | {info['event']}")
            if not a.no_render:
                time.sleep(a.delay)
        totals.append(R)
        print(f"  -> outcome: {info['outcome']}   episode return = {R:.2f}")
        if not a.no_render:
            time.sleep(2.0)
    print(f"\nMean return over {a.episodes} episodes: {np.mean(totals):.2f}")
    env.close()


if __name__ == "__main__":
    main()
