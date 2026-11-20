"""Non-learned reference policies, evaluated on the same unseen sessions as the trained agents.

    uv run python -m training.baselines
"""
from __future__ import annotations

import torch

from environment.custom_env import N_ACTIONS, OBS_SPEC

IDX = {name: i for i, (name, _, _) in enumerate(OBS_SPEC)}


class RandomPolicy(torch.nn.Module):
    """Uniform random logits: argmax picks a random action."""

    def __init__(self, seed: int = 0):
        super().__init__()
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, obs):
        return torch.rand(obs.shape[0], N_ACTIONS, generator=self.g)


class ScriptedPolicy(torch.nn.Module):
    """A hand-written trauma-informed script, written from the SafePath design, not learned.

    privacy on shared phones -> calm if very distressed -> safety screen (up to 3 tries)
    -> hand off if at risk -> build trust -> explain options -> ask preference
    -> ask consent (backing off under pressure) -> refer.
    """

    def __init__(self):
        super().__init__()
        self.screen_tries = 0

    def reset(self):
        self.screen_tries = 0

    def act(self, o) -> int:
        g = lambda k: float(o[IDX[k]])  # noqa: E731
        if g("time_frac") < 0.03:
            self.reset()
        if g("shared_dev_hint") > 0.5 and g("privacy_on") < 0.5:
            return 4                                   # ENABLE_PRIVACY
        if g("distress_est") > 0.72:
            return 5                                   # GROUNDING
        if g("safety_checked") < 0.5 and self.screen_tries < 3:
            self.screen_tries += 1
            return 3                                   # SAFETY_CHECK
        if g("risk_flag") > 0.5:
            return 8                                   # ESCALATE_HUMAN
        if g("trust_est") < 0.45:
            return 0                                   # LISTEN
        if g("info_coverage") < 0.5:
            return 1                                   # PROVIDE_INFO
        if max(g("pref_counselling"), g("pref_legal"), g("pref_medical"), g("pref_shelter")) < 0.5:
            return 2                                   # ASK_PREFERENCES
        if g("consent_given") < 0.5:
            return 0 if g("pressure") > 0.5 else 6     # LISTEN / REQUEST_CONSENT
        if g("readiness_est") < 0.55:
            return 1 if g("info_coverage") < 1 else 0  # more info / listen
        return 7                                       # OFFER_REFERRAL

    def forward(self, obs):
        out = torch.zeros(obs.shape[0], N_ACTIONS)
        for i, o in enumerate(obs.numpy()):
            out[i, self.act(o)] = 1.0
        return out


def reference_scores(n_episodes: int = 200, env_kwargs: dict | None = None) -> dict:
    from training.common import evaluate
    return {"random": evaluate(RandomPolicy(), n_episodes, env_kwargs=env_kwargs),
            "scripted": evaluate(ScriptedPolicy(), n_episodes, env_kwargs=env_kwargs)}


if __name__ == "__main__":
    for v in (1, 2):
        for name, r in reference_scores(env_kwargs={"reward_version": v}).items():
            print(f"reward v{v} {name:9s} return={r['eval_mean_return']:6.2f} success={r['success_rate']:.2f} "
                  f"risk_handled={r['risk_handled_rate']:.2f} {r['outcomes']}")
