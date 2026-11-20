"""Shared pieces for every algorithm: TorchRL env factory, networks, evaluation,
logging and model serialisation."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.envs import GymWrapper, TransformedEnv, Compose, StepCounter, RewardSum, DoubleToFloat

from environment.custom_env import SafePathEnv, OBS_DIM, N_ACTIONS

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"


def make_env(seed: int = 0, max_steps: int = 40, reward_version: int = 1) -> TransformedEnv:
    """SafePathEnv wrapped for TorchRL (one-hot actions, episode reward tracking)."""
    base = GymWrapper(SafePathEnv(max_steps=max_steps, reward_version=reward_version), device="cpu")
    env = TransformedEnv(base, Compose(DoubleToFloat(), StepCounter(), RewardSum()))
    env.set_seed(seed)
    return env


def mlp(out_dim: int, hidden: int = 128, layers: int = 2, act=nn.Tanh) -> nn.Sequential:
    mods, d = [], OBS_DIM
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), act()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------- evaluation --
@torch.no_grad()
def evaluate(net: nn.Module, n_episodes: int = 200, seed: int = 10_000, greedy: bool = True,
             env_kwargs: dict | None = None) -> dict:
    """Run the trained network on the raw gym env with unseen seeds.

    `net` maps observation -> logits or Q-values (both argmax'd when greedy).
    """
    env = SafePathEnv(**(env_kwargs or {}))
    returns, lengths, outcomes = [], [], {}
    risk_handled = risk_total = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done, R, L = False, 0.0, 0
        while not done:
            out = net(torch.as_tensor(obs).unsqueeze(0))
            if greedy:
                a = int(out.argmax(-1))
            else:
                a = int(torch.distributions.Categorical(logits=out).sample())
            obs, r, term, trunc, info = env.step(a)
            R += r
            L += 1
            done = term or trunc
        returns.append(R)
        lengths.append(L)
        oc = _bucket(info["outcome"])
        outcomes[oc] = outcomes.get(oc, 0) + 1
        if info["risk"]:
            risk_total += 1
            risk_handled += info["outcome"].startswith(("At-risk", "Escalated"))
    n = n_episodes
    return {
        "eval_mean_return": float(np.mean(returns)),
        "eval_std_return": float(np.std(returns)),
        "eval_mean_len": float(np.mean(lengths)),
        "success_rate": (outcomes.get("matched_referral", 0) + outcomes.get("human_handoff", 0)) / n,
        "matched_referral_rate": outcomes.get("matched_referral", 0) / n,
        "breach_rate": outcomes.get("privacy_breach", 0) / n,
        "dropout_rate": outcomes.get("dropout", 0) / n,
        "risk_handled_rate": risk_handled / max(1, risk_total),
        "outcomes": outcomes,
    }


def _bucket(outcome: str) -> str:
    """Turn the environment's outcome message into a short category name."""
    if outcome.startswith("Connected to a generic"):
        return "generic_referral"
    if outcome.startswith("Connected"):
        return "matched_referral"
    if outcome.startswith(("At-risk", "Escalated")):
        return "human_handoff"
    if outcome.startswith("PRIVACY"):
        return "privacy_breach"
    if outcome.startswith("She left"):
        return "dropout"
    if outcome.startswith("Session closed"):
        return "closed"
    return "timeout"


# ------------------------------------------------------------------- logging --
class RunLogger:
    """Collects one row per update for the run's CSV and prints a progress line every 10 updates."""

    def __init__(self, algo: str, run_name: str, config: dict):
        self.dir = LOG_DIR / algo
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_name}.csv"
        self.rows: list[dict] = []
        self.config = config
        self.algo, self.run_name = algo, run_name
        self.t0 = time.time()

    def log(self, **kw):
        kw["wall_s"] = round(time.time() - self.t0, 1)
        self.rows.append(kw)
        if len(self.rows) % 10 == 1:
            msg = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items())
            print(f"[{self.algo}:{self.run_name}] {msg}", flush=True)

    def finish(self, eval_res: dict):
        keys = sorted({k for r in self.rows for k in r})
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        summary = {"algo": self.algo, "run": self.run_name, "config": self.config,
                   "train_time_s": round(time.time() - self.t0, 1), **eval_res}
        with open(self.dir / f"{self.run_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{self.algo}:{self.run_name}] EVAL return={eval_res['eval_mean_return']:.2f} "
              f"success={eval_res['success_rate']:.2f} breach={eval_res['breach_rate']:.2f}", flush=True)
        return summary


# -------------------------------------------------------------- serialisation --
def save_model(net: nn.Module, kind: str, algo: str, run_name: str, config: dict, hidden: int, layers: int):
    """Save as a plain state_dict + JSON metadata so any app can load it without TorchRL."""
    d = MODEL_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), d / f"{algo}_{run_name}.pt")
    meta = {"algo": algo, "run": run_name, "hidden": hidden, "layers": layers,
            "obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "output": "q_values" if algo == "dqn" else "logits",
            "config": config}
    with open(d / f"{algo}_{run_name}.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model(path: str | Path) -> tuple[nn.Module, dict]:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    act = nn.ReLU if meta["algo"] == "dqn" else nn.Tanh
    net = mlp(meta["n_actions"], meta["hidden"], meta["layers"], act)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net, meta
