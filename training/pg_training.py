"""Policy-gradient methods on SafePath with TorchRL: REINFORCE, A2C and PPO.

    uv run python -m training.pg_training --algo ppo --run baseline
    uv run python -m training.pg_training --algo reinforce --sweep
"""
from __future__ import annotations

import argparse

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from training.common import RunLogger, evaluate, make_env, mlp, save_model

# --------------------------------------------------------------- configs ---
BASE = {
    "reinforce": dict(lr=1e-3, gamma=0.97, frames_per_batch=2_000, total_frames=300_000,
                      entropy_coef=0.01, baseline=True, normalize_returns=True,
                      hidden=128, layers=2, max_grad_norm=1.0, seed=0),
    "a2c": dict(lr=7e-4, gamma=0.97, gae_lambda=0.95, frames_per_batch=200, total_frames=300_000,
                entropy_coef=0.01, critic_coef=0.5, hidden=128, layers=2, max_grad_norm=0.5,
                normalize_advantage=False, seed=0),
    "ppo": dict(lr=3e-4, gamma=0.97, gae_lambda=0.95, frames_per_batch=2_000, total_frames=300_000,
                epochs=6, minibatch=256, clip_eps=0.2, entropy_coef=0.01, critic_coef=0.5,
                hidden=128, layers=2, max_grad_norm=0.5, normalize_advantage=True, seed=0),
}

SWEEPS = {
    # REINFORCE: lr, gamma, episodes per update, entropy, baseline / return normalisation
    "reinforce": {
        "baseline":       {},
        "lr_high":        dict(lr=5e-3),
        "lr_low":         dict(lr=2e-4),
        "gamma_low":      dict(gamma=0.90),
        "gamma_high":     dict(gamma=0.995),
        "batch_small":    dict(frames_per_batch=500),
        "batch_large":    dict(frames_per_batch=6_000),
        "no_entropy":     dict(entropy_coef=0.0),
        "entropy_high":   dict(entropy_coef=0.05),
        "no_baseline":    dict(baseline=False, normalize_returns=False),
        "small_net":      dict(hidden=64, layers=1),
        "batch_small_v2": dict(frames_per_batch=500, reward_version=2),
    },
    # A2C: lr, rollout length (n-step), GAE lambda, entropy, value coefficient
    "a2c": {
        "baseline":       {},
        "lr_high":        dict(lr=3e-3),
        "lr_low":         dict(lr=1e-4),
        "gamma_low":      dict(gamma=0.90),
        "rollout_short":  dict(frames_per_batch=40),
        "rollout_long":   dict(frames_per_batch=1_000),
        "lambda_td":      dict(gae_lambda=0.0),
        "lambda_mc":      dict(gae_lambda=1.0),
        "entropy_high":   dict(entropy_coef=0.05),
        "no_entropy":     dict(entropy_coef=0.0),
        "critic_high":    dict(critic_coef=1.0, normalize_advantage=True),
        "lambda_mc_v2":   dict(gae_lambda=1.0, reward_version=2),
    },
    # PPO: lr, clip, epochs, minibatch, rollout length, GAE lambda, entropy, value coef
    "ppo": {
        "baseline":       {},
        "lr_high":        dict(lr=1e-3),
        "lr_low":         dict(lr=5e-5),
        "clip_tight":     dict(clip_eps=0.05),
        "clip_wide":      dict(clip_eps=0.4),
        "epochs_many":    dict(epochs=15),
        "epochs_one":     dict(epochs=1),
        "rollout_short":  dict(frames_per_batch=500, minibatch=64),
        "lambda_low":     dict(gae_lambda=0.5),
        "entropy_high":   dict(entropy_coef=0.05),
        "no_entropy":     dict(entropy_coef=0.0),
        "critic_low":     dict(critic_coef=0.1),
        "epochs_many_v2": dict(epochs=15, reward_version=2),
    },
}


def build_actor_critic(env, cfg):
    n_act = env.action_spec.shape[-1]
    policy_net = mlp(n_act, cfg["hidden"], cfg["layers"], torch.nn.Tanh)
    actor = ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], distribution_class=OneHotCategorical, spec=env.action_spec,
        return_log_prob=True,
    )
    critic = ValueOperator(mlp(1, cfg["hidden"], cfg["layers"], torch.nn.Tanh), in_keys=["observation"])
    return policy_net, actor, critic


def discounted_returns(reward, done, gamma):
    """Reward-to-go per trajectory inside a flat batch (resets at episode ends)."""
    ret = torch.zeros_like(reward)
    running = torch.zeros(1)
    for t in range(reward.shape[0] - 1, -1, -1):
        if done[t]:
            running = torch.zeros(1)
        running = reward[t] + gamma * running
        ret[t] = running
    return ret


def _entropy(ld, actor, td) -> float:
    """Policy entropy for logging. TorchRL omits it from the loss output when entropy_coeff=0."""
    if "entropy" in ld.keys():
        return ld["entropy"].item()
    with torch.no_grad():
        return actor.get_dist(td.select("observation").clone()).entropy().mean().item()


def train(algo: str, run_name: str, overrides: dict) -> dict:
    cfg = {**BASE[algo], **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env(cfg["seed"], reward_version=cfg.get("reward_version", 1))
    policy_net, actor, critic = build_actor_critic(env, cfg)
    collector = Collector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"])
    log = RunLogger(algo, run_name, cfg)

    if algo == "ppo":
        adv = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
        loss_mod = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_eps"], entropy_coeff=cfg["entropy_coef"],
                               critic_coeff=cfg["critic_coef"], normalize_advantage=cfg["normalize_advantage"])
        rb = ReplayBuffer(storage=LazyTensorStorage(cfg["frames_per_batch"]),
                          sampler=SamplerWithoutReplacement(), batch_size=cfg["minibatch"])
    elif algo == "a2c":
        adv = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
        loss_mod = A2CLoss(actor, critic, entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["critic_coef"])
    params = list(policy_net.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    frames, ep_returns = 0, []
    for batch in collector:
        frames += batch.numel()
        batch = batch.reshape(-1)
        done = batch["next", "done"].squeeze(-1)
        ep_returns += batch["next", "episode_reward"][done].flatten().tolist()
        stats = {}

        if algo == "reinforce":
            rew = batch["next", "reward"].squeeze(-1)
            # the final partial trajectory is treated as cut (bootstrap 0)
            G = discounted_returns(rew, done, cfg["gamma"])
            if cfg["baseline"]:
                # learned state-value baseline (REINFORCE with baseline)
                v = critic(batch.select("observation").clone())["state_value"].squeeze(-1)
                advantage = G - v.detach()
                critic_loss = 0.5 * (v - G).pow(2).mean()
            else:
                advantage, critic_loss = G, torch.zeros(())
            if cfg["normalize_returns"]:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            dist = actor.get_dist(batch.select("observation").clone())
            logp = dist.log_prob(batch["action"])
            entropy = dist.entropy().mean()
            policy_loss = -(logp * advantage).mean()
            total = policy_loss - cfg["entropy_coef"] * entropy + critic_loss
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            opt.step()
            stats = dict(loss_policy=policy_loss.item(), loss_critic=float(critic_loss), entropy=entropy.item())

        elif algo == "a2c":
            # A2C: one update per short rollout, with GAE advantages from the critic
            with torch.no_grad():
                adv(batch)
            if cfg["normalize_advantage"]:
                a = batch["advantage"]
                batch["advantage"] = (a - a.mean()) / (a.std() + 1e-8)
            ld = loss_mod(batch)
            total = ld["loss_objective"] + ld["loss_critic"] + ld.get("loss_entropy", torch.zeros(()))
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            opt.step()
            stats = dict(loss_policy=ld["loss_objective"].item(), loss_critic=ld["loss_critic"].item(),
                         entropy=_entropy(ld, actor, batch))

        else:
            # PPO: recompute GAE each epoch, then several passes of shuffled mini-batches
            # through the clipped loss, all on the same rollout
            acc = {"loss_policy": 0.0, "loss_critic": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
            n = 0
            for _ in range(cfg["epochs"]):
                with torch.no_grad():
                    adv(batch)
                rb.empty()
                rb.extend(batch)
                for _ in range(max(1, cfg["frames_per_batch"] // cfg["minibatch"])):
                    mb = rb.sample()
                    ld = loss_mod(mb)
                    total = ld["loss_objective"] + ld["loss_critic"] + ld.get("loss_entropy", torch.zeros(()))
                    opt.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                    opt.step()
                    acc["loss_policy"] += ld["loss_objective"].item()
                    acc["loss_critic"] += ld["loss_critic"].item()
                    acc["entropy"] += _entropy(ld, actor, mb)
                    if "clip_fraction" in ld.keys():
                        acc["clip_fraction"] += ld["clip_fraction"].item()
                    n += 1
            stats = {k: v / n for k, v in acc.items()}

        collector.update_policy_weights_()  # the collector must act with the updated policy
        recent = ep_returns[-50:]
        log.log(frames=frames, ep_return=sum(recent) / max(1, len(recent)), episodes=len(ep_returns), **stats)
    collector.shutdown()

    res = evaluate(policy_net, greedy=True, env_kwargs={"reward_version": cfg.get("reward_version", 1)})
    save_model(policy_net, "pg", algo, run_name, cfg, cfg["hidden"], cfg["layers"])
    return log.finish(res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["reinforce", "a2c", "ppo"])
    p.add_argument("--run", default="baseline")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--frames", type=int, default=None)
    a = p.parse_args()
    torch.set_num_threads(1)
    extra = {"total_frames": a.frames} if a.frames else {}
    sw = SWEEPS[a.algo]
    for name in (sw if a.sweep else [a.run]):
        train(a.algo, name, {**sw[name], **extra})
