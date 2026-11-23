"""Record support sessions played by trained agents so the 3D web view can replay them.

    uv run python -m training.export_replay                  # best safe agent of each algorithm + baselines
    uv run python -m training.export_replay --model models/pg/ppo_baseline.pt --name ppo

Writes web/replays/<name>.json and web/replays/index.json. Every agent plays the same four sessions,
one per survivor profile, so the agents can be compared on identical people.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import ACTIONS, SERVICES, SafePathEnv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "replays"

# The first seed from 2026 of each profile type (risk x shared device). 2028 stands in for the
# plain profile because 2027 is one where REINFORCE refers to a generic service; both are typical.
SESSIONS = [
    (2028, "Ready to talk", "No immediate danger, private phone"),
    (2026, "Shared phone", "Someone else uses her phone"),
    (2035, "In danger", "Immediate danger, private phone"),
    (2034, "In danger, shared phone", "Immediate danger and a shared phone"),
]


def stage(env: SafePathEnv, start: dict) -> float:
    """Position along the pathway (0 safe space .. 4 support), counted from where she started.

    Safe space -> Trust: making her safe (discreet mode, safety screen) and trust gained.
    Trust -> Options: options explained and her preference heard.  Options -> Consent: consent,
    or readiness gained on the way to it.
    """
    if env.outcome.startswith(("Connected", "At-risk", "Escalated")):
        return 4.0
    s = min(1.0, 0.4 * env.privacy_on + 0.4 * env.safety_checked + max(0.0, env.trust - start["trust"]) / 0.15)
    s += min(1.0, env.info / 4 + 0.5 * env.pref_known)
    s += 1.0 if env.consent else min(0.6, 2 * max(0.0, env.readiness - start["readiness"]))
    return round(min(s, 3.6), 3)


def snapshot(env: SafePathEnv, start: dict) -> dict:
    return {"distress": round(env.distress, 3), "trust": round(env.trust, 3),
            "readiness": round(env.readiness, 3), "info": env.info, "consent": env.consent,
            "safety_checked": env.safety_checked, "risk_flag": bool(env.safety_checked and env.risk),
            "privacy_on": env.privacy_on, "exposure": round(env.exposure, 3),
            "pressure": round(env.pressure, 3), "pref_known": env.pref_known,
            "outcome": env.outcome, "stage": stage(env, start)}


@torch.no_grad()
def record_session(policy, seed: int, title: str, blurb: str) -> dict:
    env = SafePathEnv()
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    if hasattr(policy, "reset"):
        policy.reset()
    start = {"trust": env.trust, "readiness": env.readiness}
    steps = [{"t": 0, "action": None, "reward": 0.0, "total": 0.0, "event": "Session opened", **snapshot(env, start)}]
    total, done = 0.0, False
    while not done:
        a = policy(obs, rng)
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
        steps.append({"t": env.t, "action": a, "reward": round(r, 3), "total": round(total, 3),
                      "event": info["event"], **snapshot(env, start)})
    return {"seed": seed, "title": title, "blurb": blurb,
            "profile": {"risk": env.risk, "shared_device": env.shared_device,
                        "preference": SERVICES[env.preference], "preference_idx": env.preference},
            "outcome": info["outcome"], "return": round(total, 2), "steps": steps}


def net_policy(net):
    return lambda obs, rng: int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))


class _Scripted:
    def __init__(self):
        from training.baselines import ScriptedPolicy
        self.p = ScriptedPolicy()

    def reset(self):
        self.p.reset()

    def __call__(self, obs, rng):
        return self.p.act(obs)


def main():
    from training.common import load_model
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--name", default=None)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    if a.model:
        net, meta = load_model(a.model)
        jobs.append((net_policy(net), a.name or meta["algo"], f"{meta['algo'].upper()} ({meta['run']})", ""))
    else:
        sel = json.loads((ROOT / "assets" / "selected_runs.json").read_text())
        for algo, run in sel["safe"].items():
            kind = "dqn" if algo == "dqn" else "pg"
            net, _ = load_model(ROOT / "models" / kind / f"{algo}_{run}.pt")
            jobs.append((net_policy(net), algo, f"{algo.upper()} ({run})", "best safe run"))
        # the top-return PPO run from the sweep: it learned to skip the safety screen (reward hacking)
        hack = sel["top_return"]["ppo"]
        net, _ = load_model(ROOT / "models" / "pg" / f"ppo_{hack}.pt")
        jobs.append((net_policy(net), "ppo_hack", f"PPO ({hack}), reward hacker", "top return, skips the safety screen"))
        jobs.append((_Scripted(), "scripted", "Hand-written script", "trauma-informed rules, not learned"))
        jobs.append((lambda obs, rng: int(rng.integers(len(ACTIONS))), "random", "Random agent", "untrained"))

    for policy, name, label, note in jobs:
        sessions = [record_session(policy, s, t, b) for s, t, b in SESSIONS]
        rec = {"name": name, "label": label, "note": note, "actions": ACTIONS, "services": SERVICES,
               "max_steps": 40, "sessions": sessions}
        (OUT / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":")))
        print(f"{label:34s} " + "  ".join(f"{s['return']:6.2f} {s['outcome'][:28]:28s}" for s in sessions))

    order = ["ppo", "dqn", "a2c", "reinforce", "ppo_hack", "scripted", "random"]
    files = sorted((f.stem for f in OUT.glob("*.json") if f.name != "index.json"),
                   key=lambda n: order.index(n) if n in order else 99)
    (OUT / "index.json").write_text(json.dumps(
        [{"label": json.loads((OUT / f"{n}.json").read_text())["label"], "file": f"{n}.json"} for n in files]))


if __name__ == "__main__":
    main()
