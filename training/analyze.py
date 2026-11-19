"""Build every report figure and table from logs/.

    uv run python -m training.analyze

Outputs -> assets/figures/*.png and assets/results_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import LOG_DIR, MODEL_DIR, evaluate, load_model

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "assets" / "figures"
ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
# validated categorical palette (fixed order, light mode)
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "a2c": "#1baf7a", "ppo": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "lines.linewidth": 2, "legend.frameon": False,
})

# columns shown in the per-algorithm tables (the hyperparameters that are swept)
TABLE_COLS = {
    "dqn": ["lr", "gamma", "batch_size", "buffer_size", "eps_frac", "eps_end", "target_update", "tau",
            "double_dqn", "prioritized", "hidden"],
    "reinforce": ["lr", "gamma", "frames_per_batch", "entropy_coef", "baseline", "normalize_returns", "hidden", "layers"],
    "a2c": ["lr", "gamma", "frames_per_batch", "gae_lambda", "entropy_coef", "critic_coef", "normalize_advantage"],
    "ppo": ["lr", "gamma", "frames_per_batch", "minibatch", "epochs", "clip_eps", "gae_lambda", "entropy_coef", "critic_coef"],
}


def load():
    """Sweep runs (reward v1) and the reward-v2 re-runs, kept apart."""
    summaries, v2, curves = [], [], {}
    for algo in ALGOS:
        for f in sorted((LOG_DIR / algo).glob("*_summary.json")):
            s = json.loads(f.read_text())
            (v2 if s["config"].get("reward_version", 1) == 2 else summaries).append(s)
            curves[(algo, s["run"])] = pd.read_csv(LOG_DIR / algo / f"{s['run']}.csv")
    return summaries, v2, curves


def smooth(y, k=9):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return pd.Series(y).rolling(k, min_periods=1).mean().to_numpy()


# Safety rule for model selection: an agent only qualifies as "best" if it hands at least
# 90% of at-risk survivors to a human. Without it, selection by return picks agents that
# learned to skip the safety screen under the v1 reward (see the report).
SAFE_MIN = 0.9


def is_safe(s):
    return s["risk_handled_rate"] >= SAFE_MIN


def best_runs(summaries, safe_only=True):
    best = {}
    for s in summaries:
        if safe_only and not is_safe(s):
            continue
        if s["algo"] not in best or s["eval_mean_return"] > best[s["algo"]]["eval_mean_return"]:
            best[s["algo"]] = s
    return best


def frames_to_converge(df, final_frac=0.9):
    """First frame where smoothed return reaches 90% of its final level (relative to start)."""
    y = smooth(df["ep_return"], 15)
    start, end = y[0], np.mean(y[-10:])
    if end <= start:
        return np.nan
    thr = start + final_frac * (end - start)
    idx = np.argmax(y >= thr)
    return int(df["frames"].iloc[idx])


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK, fontweight="bold")
    ax.plot([x], [y], "o", ms=5, color=color, mec="#fcfcfb", mew=2)


def fig_comparative(best, curves, refs):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for algo in ALGOS:
        if algo not in best:
            continue
        df = curves[(algo, best[algo]["run"])]
        y = smooth(df["ep_return"], 15)
        ax.plot(df["frames"], y, color=COLOR[algo], label=f"{LABEL[algo]} ({best[algo]['run']})")
        label_end(ax, df["frames"].iloc[-1], y[-1], LABEL[algo], COLOR[algo])
    sc, rnd = refs["scripted"]["eval_mean_return"], refs["random"]["eval_mean_return"]
    ax.axhline(sc, color=INK2, lw=1, ls="--")
    ax.text(ax.get_xlim()[0], sc - 0.9, f" hand-written script, evaluated ({sc:.1f})", color=INK2, fontsize=8)
    ax.axhline(rnd, color=INK2, lw=1, ls=":")
    ax.text(ax.get_xlim()[0], rnd + 0.3, f" random agent, evaluated ({rnd:.1f})", color=INK2, fontsize=8)
    ax.set(xlabel="Environment steps (frames)", ylabel="Mean episode return (last 50 episodes)",
           title="Fig. 1  Training return of the best configuration of each algorithm")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_comparative_rewards.png", dpi=160)
    plt.close(fig)


def fig_sweeps(summaries, curves, best):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        runs = [s for s in summaries if s["algo"] == algo]
        for s in runs:
            df = curves[(algo, s["run"])]
            is_best = algo in best and s["run"] == best[algo]["run"]
            ax.plot(df["frames"], smooth(df["ep_return"], 15),
                    color=COLOR[algo] if is_best else "#b9b8b2", lw=2.2 if is_best else 1,
                    zorder=3 if is_best else 1, label=f"best: {s['run']}" if is_best else None)
        ax.set_title(f"{LABEL[algo]} — {len(runs)} configurations")
        ax.set_xlabel("frames")
        ax.legend(loc="lower right", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean episode return")
    fig.suptitle("Fig. 2  Convergence across every hyperparameter configuration (grey) and the best run",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_sweep_convergence.png", dpi=160)
    plt.close(fig)


def fig_dqn_loss(summaries, curves, best):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    show = [best["dqn"]["run"]] + [r for r in ["lr_high", "vanilla_dqn", "soft_target"] if r != best["dqn"]["run"]]
    shades = [COLOR["dqn"], "#86b6ef", "#184f95", "#5598e7"]
    for run, col in zip(show, shades):
        if ("dqn", run) not in curves:
            continue
        df = curves[("dqn", run)]
        y = smooth(df["loss"], 15)
        ax.plot(df["frames"], y, color=col, label=run, lw=2.2 if run == show[0] else 1.4)
    ax.set(xlabel="frames", ylabel="Smooth-L1 TD loss",
           title="Fig. 3  DQN objective (TD loss) — best run vs. selected variants")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_dqn_loss.png", dpi=160)
    plt.close(fig)


def fig_entropy(curves, best, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for algo in ["reinforce", "a2c", "ppo"]:
        if algo not in best:
            continue
        df = curves[(algo, best[algo]["run"])]
        y = smooth(df["entropy"], 15)
        ax.plot(df["frames"], y, color=COLOR[algo], label=LABEL[algo])
        label_end(ax, df["frames"].iloc[-1], y[-1], LABEL[algo], COLOR[algo])
    ax.axhline(np.log(10), color=INK2, lw=1, ls=":")
    ax.text(0, np.log(10) - 0.12, " uniform policy ln(10)", color=INK2, fontsize=8)
    ax.set(xlabel="frames", ylabel="policy entropy (nats)", title="Best runs")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    for run, ls in [("baseline", "-"), ("no_entropy", "--"), ("entropy_high", ":")]:
        if ("ppo", run) in curves:
            df = curves[("ppo", run)]
            ax.plot(df["frames"], smooth(df["entropy"], 15), color=COLOR["ppo"], ls=ls, label=f"PPO {run}")
    ax.set(xlabel="frames", title="Effect of the entropy coefficient (PPO)")
    ax.legend(fontsize=8)
    fig.suptitle("Fig. 4  Policy-gradient entropy during training", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_pg_entropy.png", dpi=160)
    plt.close(fig)


GEN_SETTINGS = {
    "unseen seeds": {},
    "noisier signals (σ=0.15)": {"obs_noise": 0.15},
    "high-risk population (50%)": {"risk_prob": 0.5},
    "mostly shared phones (80%)": {"shared_prob": 0.8},
}


def generalization(best):
    rows = []
    for algo, s in best.items():
        kind = "dqn" if algo == "dqn" else "pg"
        net, _ = load_model(MODEL_DIR / kind / f"{algo}_{s['run']}.pt")
        for name, kw in GEN_SETTINGS.items():
            r = evaluate(net, n_episodes=500, seed=50_000, env_kwargs=kw)
            rows.append({"algo": algo, "setting": name, **{k: v for k, v in r.items() if k != "outcomes"},
                         **{f"oc_{k}": v / 500 for k, v in r["outcomes"].items()}})
    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv(ROOT / "assets" / "generalization.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    settings = list(GEN_SETTINGS)
    w = 0.2
    x = np.arange(len(settings))
    for i, algo in enumerate([a for a in ALGOS if a in best]):
        vals = [df[(df.algo == algo) & (df.setting == st)]["eval_mean_return"].item() for st in settings]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=COLOR[algo], label=LABEL[algo])
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in settings], fontsize=8)
    ax.set(ylabel="mean return (500 episodes)", title="Return under distribution shift")
    ax.legend(fontsize=8, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    ax = axes[1]
    base = df[df.setting == "unseen seeds"].set_index("algo")
    cats = [("oc_matched_referral", "matched referral", "#1baf7a"),
            ("oc_human_handoff", "human hand-off", "#2a78d6"),
            ("oc_generic_referral", "generic referral", "#86b6ef"),
            ("oc_closed", "closed early", "#b9b8b2"),
            ("oc_timeout", "timed out", "#e4e3df"),
            ("oc_dropout", "drop-out", "#eb6834"),
            ("oc_privacy_breach", "privacy breach", "#e34948")]
    algos = [a for a in ALGOS if a in best]
    left = np.zeros(len(algos))
    for col, name, c in cats:
        v = np.array([base.loc[a, col] if col in base.columns else 0 for a in algos])
        ax.barh([LABEL[a] for a in algos], v, left=left, color=c, label=name, edgecolor="#fcfcfb", linewidth=2)
        for j, (l, vv) in enumerate(zip(left, v)):
            if vv > 0.08:
                ax.text(l + vv / 2, j, f"{vv:.0%}", ha="center", va="center", fontsize=8, color=INK)
        left += v
    ax.set_xlim(0, 1)
    ax.set(xlabel="share of 500 unseen sessions", title="How sessions end (best agents)")
    ax.legend(fontsize=7, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.invert_yaxis()
    fig.suptitle("Fig. 5  Generalisation: best agents on unseen users and shifted populations",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_generalization.png", dpi=160)
    plt.close(fig)
    return df


def fig_reward_v2(summaries, v2):
    """Same hyperparameters, reward v1 vs v2: does penalising missed danger stop the hacking?"""
    by = {(s["algo"], s["run"]): s for s in summaries}
    rows = []
    for s2 in v2:
        s1 = by.get((s2["algo"], s2["run"].removesuffix("_v2")))
        if not s1:
            continue
        for ver, s in (("v1", s1), ("v2", s2)):
            rows.append({"algo": s2["algo"], "run": s1["run"], "reward": ver,
                         "risk_handled": s["risk_handled_rate"], "success": s["success_rate"],
                         "matched": s["matched_referral_rate"], "dropout": s["dropout_rate"],
                         "breach": s["breach_rate"], "return": s["eval_mean_return"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "assets" / "reward_v1_vs_v2.csv", index=False)
    pairs = df.drop_duplicates(["algo", "run"])[["algo", "run"]].values.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(pairs))
    for ax, col, title in [(axes[0], "risk_handled", "At-risk survivors handed to a human"),
                           (axes[1], "success", "Sessions ending in a good outcome")]:
        for j, ver in enumerate(["v1", "v2"]):
            vals = [df[(df.algo == a) & (df.run == r) & (df.reward == ver)][col].item() for a, r in pairs]
            cols = ["#b9b8b2" if ver == "v1" else COLOR[a] for a, _ in pairs]
            bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=cols,
                          label="reward v1 (grey)" if ver == "v1" else "reward v2: missed danger −8 (colour)")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
        ax.set_xticks(x, [f"{LABEL[a]}\n{r}" for a, r in pairs], fontsize=8)
        ax.set_ylim(0, 1.18)
        ax.set_title(title)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Fig. 6  Same hyperparameters, reward v1 vs v2", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_reward_v1_vs_v2.png", dpi=160)
    plt.close(fig)
    print(df.to_string(index=False))


def tables(summaries, curves):
    out = ["# Hyperparameter results\n",
           "All runs: same SafePath environment, same seed (0), evaluation = 200 unseen sessions "
           "(seeds 10000-10199), greedy policy. Best = highest return among runs with "
           f"risk→human ≥ {SAFE_MIN:.0%}.\n"]
    for algo in ALGOS:
        runs = [s for s in summaries if s["algo"] == algo]
        if not runs:
            continue
        cols = TABLE_COLS[algo]
        out.append(f"\n## {LABEL[algo]} ({len(runs)} runs)\n")
        out.append("| # | run | " + " | ".join(cols) +
                   " | frames | eval return | success | risk→human | breach | drop-out | conv. frames |")
        out.append("|" + "---|" * (len(cols) + 9))
        for i, s in enumerate(sorted(runs, key=lambda s: -s["eval_mean_return"]), 1):
            c = s["config"]
            conv = frames_to_converge(curves[(algo, s["run"])])
            vals = [str(c.get(k, "")) for k in cols]
            out.append(f"| {i} | {s['run']} | " + " | ".join(vals) +
                       f" | {c['total_frames']//1000}k | {s['eval_mean_return']:.2f} ± {s['eval_std_return']:.1f} | "
                       f"{s['success_rate']:.0%} | {s['risk_handled_rate']:.0%} | "
                       f"{s['breach_rate']:.0%} | {s['dropout_rate']:.0%} | "
                       f"{'—' if np.isnan(conv) else f'{conv//1000}k'} |")
    (ROOT / "assets" / "results_tables.md").write_text("\n".join(out), encoding="utf-8")


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    from training.baselines import reference_scores
    summaries, v2, curves = load()
    best = best_runs(summaries)
    top = best_runs(summaries, safe_only=False)
    print("best (safe) runs:", {a: (s["run"], round(s["eval_mean_return"], 2)) for a, s in best.items()})
    print("top by return   :", {a: (s["run"], round(s["eval_mean_return"], 2), s["risk_handled_rate"])
                                for a, s in top.items()})
    refs = reference_scores()
    (ROOT / "assets" / "reference_policies.json").write_text(json.dumps(refs, indent=2))
    (ROOT / "assets" / "selected_runs.json").write_text(json.dumps(
        {"safe": {a: s["run"] for a, s in best.items()}, "top_return": {a: s["run"] for a, s in top.items()}},
        indent=2))
    fig_comparative(best, curves, refs)
    fig_sweeps(summaries, curves, best)
    if "dqn" in best:
        fig_dqn_loss(summaries, curves, best)
    fig_entropy(curves, best, summaries)
    tables(summaries, curves)
    if v2:
        fig_reward_v2(summaries, v2)
    g = generalization(best)
    print(g[["algo", "setting", "eval_mean_return", "success_rate", "breach_rate", "risk_handled_rate"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
