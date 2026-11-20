"""SafePath support-navigation environment.

The agent is the conversational support assistant inside the SafePath app. It
never decides for the survivor: every action is an *offer* (listen, inform, ask,
protect privacy, request consent, offer a referral). A simulated survivor with a
hidden internal state responds stochastically, and only she can accept.

The agent is rewarded when the survivor connects to the support she chose, with
consent, without a privacy leak, and when safety risk is routed to a human
professional. It is penalised for pushing without consent, overwhelming a
distressed user, privacy breaches and drop-out.

Hidden state  (true, never observed directly)
    distress, trust, readiness, risk (immediate danger), shared_device,
    preference (counselling / legal / medical / shelter)
Observation   16 floats (see OBS_SPEC): noisy session signals the real app
    would compute from check-ins, engagement and explicit answers.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------- actions ---
ACTIONS = [
    "LISTEN",            # 0 empathetic reflection
    "PROVIDE_INFO",      # 1 explain one support option
    "ASK_PREFERENCES",   # 2 ask what kind of help she wants
    "SAFETY_CHECK",      # 3 gentle safety screening question
    "ENABLE_PRIVACY",    # 4 discreet mode + quick-exit + auto-clear history
    "GROUNDING",         # 5 breathing / grounding exercise
    "REQUEST_CONSENT",   # 6 ask permission to share details with a service
    "OFFER_REFERRAL",    # 7 offer referral to her preferred service
    "ESCALATE_HUMAN",    # 8 hand off live to a trained human counsellor
    "SAVE_AND_CLOSE",    # 9 save resources discreetly, end session gently
]
N_ACTIONS = len(ACTIONS)
SERVICES = ["counselling", "legal aid", "medical care", "safe shelter"]

# ------------------------------------------------------------ observation ---
# (name, range, meaning / real-world source)
OBS_SPEC = [
    ("distress_est",     "[0,1]", "mood check-in slider + typing-pause signal, noisy"),
    ("trust_est",        "[0,1]", "engagement proxy (reply length, latency), noisy"),
    ("readiness_est",    "[0,1]", "stage-of-change answer, noisy"),
    ("info_coverage",    "[0,1]", "fraction of the 4 support options explained"),
    ("consent_given",    "{0,1}", "explicit consent button pressed"),
    ("safety_checked",   "{0,1}", "safety screen has been completed"),
    ("risk_flag",        "{0,1}", "safety screen reported immediate danger"),
    ("privacy_on",       "{0,1}", "discreet mode / quick-exit active"),
    ("shared_dev_hint",  "{0,1}", "device signal: multiple profiles / unlocked by other"),
    ("privacy_exposure", "[0,1]", "cumulative unprotected time on a shared device"),
    ("pressure",         "[0,1]", "recent directive actions without consent"),
    ("time_frac",        "[0,1]", "step / max_steps"),
    ("pref_counselling", "{0,1}", "preference revealed: counselling"),
    ("pref_legal",       "{0,1}", "preference revealed: legal aid"),
    ("pref_medical",     "{0,1}", "preference revealed: medical care"),
    ("pref_shelter",     "{0,1}", "preference revealed: safe shelter"),
]
OBS_DIM = len(OBS_SPEC)


class SafePathEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, max_steps: int = 40, render_mode: str | None = None,
                 obs_noise: float = 0.08, risk_prob: float = 0.25, shared_prob: float = 0.40,
                 reward_version: int = 1):
        super().__init__()
        # v1: the original reward used for the 47-run sweep.
        # v2: also penalises ending a session (referral or close) with an at-risk survivor who was
        #     never handed to a human. v1 had no such penalty, so the top agents learned to skip
        #     the safety screen entirely (reward hacking, see the report).
        self.reward_version = reward_version
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.obs_noise = obs_noise
        self.risk_prob = risk_prob
        self.shared_prob = shared_prob
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._renderer = None
        self.last_event = ""

    # ----------------------------------------------------------- helpers ---
    @staticmethod
    def _clip(x):
        return float(np.clip(x, 0.0, 1.0))

    def _noisy(self, x):
        return self._clip(x + self.np_random.normal(0.0, self.obs_noise))

    def _obs(self):
        pref = np.zeros(4, dtype=np.float32)
        if self.pref_known:
            pref[self.preference] = 1.0
        o = np.array([
            self._noisy(self.distress), self._noisy(self.trust), self._noisy(self.readiness),
            self.info / 4.0, float(self.consent), float(self.safety_checked),
            float(self.safety_checked and self.risk), float(self.privacy_on),
            float(self.shared_hint), self.exposure, self.pressure,
            self.t / self.max_steps, *pref,
        ], dtype=np.float32)
        return o

    def _info(self):
        return {
            "distress": self.distress, "trust": self.trust, "readiness": self.readiness,
            "risk": self.risk, "shared_device": self.shared_device,
            "preference": SERVICES[self.preference], "outcome": self.outcome,
            "event": self.last_event,
        }

    # ------------------------------------------------------------- reset ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.distress = r.uniform(0.35, 0.95)
        self.trust = r.uniform(0.10, 0.45)
        self.readiness = r.uniform(0.10, 0.50)
        self.risk = bool(r.random() < self.risk_prob)   # default 1 in 4 in immediate danger
        self.shared_device = bool(r.random() < self.shared_prob)  # phone shared with others
        # the device hint is right 80% of the time
        self.shared_hint = self.shared_device if r.random() < 0.8 else not self.shared_device
        self.preference = int(r.integers(4))
        self.pref_known = False
        self.info = 0
        self.consent = False
        self.safety_checked = False
        self.privacy_on = False
        self.exposure = 0.0
        self.pressure = 0.0
        self.t = 0
        self.outcome = "in progress"
        self.last_event = "Session opened"
        self.last_action = None
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    # -------------------------------------------------------------- step ---
    def step(self, action):
        a = int(action)
        r = self.np_random
        reward = -0.05             # time cost: she is in a hard moment, be efficient
        terminated = False
        self.t += 1
        self.last_action = a
        directive = False
        ev = ACTIONS[a]

        if a == 0:  # LISTEN
            self.trust += 0.10 * (1 - self.trust)
            self.distress -= 0.04
            self.readiness += 0.03
            ev = "Listened and reflected her feelings"
        elif a == 1:  # PROVIDE_INFO
            if self.distress > 0.75:
                self.trust -= 0.05
                self.distress += 0.03
                ev = "Info while highly distressed: overwhelmed"
            elif self.info < 4:
                self.info += 1
                self.readiness += 0.08
                reward += 0.3
                ev = f"Explained option {self.info}/4"
            else:
                ev = "Repeated information (no new value)"
        elif a == 2:  # ASK_PREFERENCES
            if self.pref_known:
                ev = "Asked again (already known)"
                self.trust -= 0.02
            elif r.random() < self.trust + 0.2:
                self.pref_known = True
                reward += 0.5
                ev = f"She prefers {SERVICES[self.preference]}"
            else:
                ev = "Not ready to say what she wants"
        elif a == 3:  # SAFETY_CHECK
            if self.safety_checked:
                self.trust -= 0.03
                ev = "Repeated safety question"
            elif r.random() < 0.35 + self.trust:
                self.safety_checked = True
                reward += 0.5
                ev = "Safety screen: IMMEDIATE RISK" if self.risk else "Safety screen: no immediate risk"
            else:
                self.trust -= 0.04
                ev = "Safety question felt intrusive, declined"
        elif a == 4:  # ENABLE_PRIVACY
            if not self.privacy_on:
                self.privacy_on = True
                self.trust += 0.05
                reward += 0.2 if self.shared_device else 0.0
                ev = "Discreet mode + quick-exit enabled"
            else:
                ev = "Privacy already on"
        elif a == 5:  # GROUNDING
            self.distress -= 0.15
            self.trust += 0.02
            ev = "Grounding / breathing exercise"
        elif a == 6:  # REQUEST_CONSENT
            directive = True
            if self.consent:
                ev = "Consent already given"
            elif r.random() < 0.6 * self.trust + 0.4 * self.readiness:
                self.consent = True
                self.pressure = 0.0
                directive = False
                reward += 0.5
                ev = "She consented to a referral"
            else:
                self.trust -= 0.04
                ev = "She declined consent for now"
        elif a == 7:  # OFFER_REFERRAL
            if not self.consent:
                directive = True
                reward -= 1.0
                self.trust -= 0.12
                ev = "Referral without consent: agency violated"
            elif self.readiness > 0.55 and r.random() < 0.5 + 0.5 * self.trust:
                matched = self.pref_known
                reward += 10.0 if matched else 4.0
                reward += 1.5 if (self.privacy_on or not self.shared_device) else 0.0
                reward += 1.0 * (self.info / 4)
                terminated = True
                self.outcome = ("Connected to " + SERVICES[self.preference]) if matched \
                    else "Connected to a generic service"
                ev = self.outcome
            else:
                self.trust -= 0.03
                ev = "She is not ready to be referred yet"
        elif a == 8:  # ESCALATE_HUMAN
            if self.risk and self.safety_checked:
                reward += 10.0 + (1.5 if self.privacy_on or not self.shared_device else 0.0)
                terminated = True
                self.outcome = "At-risk survivor handed to human counsellor"
                ev = self.outcome
            elif self.risk:
                # escalation without screening still helps but costs her trust
                reward += 5.0
                terminated = True
                self.outcome = "Escalated without screening"
                ev = self.outcome
            else:
                reward -= 1.0
                self.trust -= 0.08
                directive = True
                ev = "Unneeded escalation felt like being passed on"
        elif a == 9:  # SAVE_AND_CLOSE
            terminated = True
            base = 1.0 + 1.0 * (self.info / 4) + (0.5 if self.privacy_on else 0.0)
            if self.risk:
                base -= 6.0           # closed on someone in danger
            reward += base
            self.outcome = "Session closed with saved resources"
            ev = self.outcome

        # --- pressure / agency dynamics --------------------------------------
        if directive:
            self.pressure = min(1.0, self.pressure + 0.34)
            if self.pressure >= 0.99:
                self.trust -= 0.10
                reward -= 0.5
        else:
            self.pressure = max(0.0, self.pressure - 0.15)

        # --- natural drift -----------------------------------------------------
        self.distress += 0.01 + r.normal(0, 0.02)
        self.distress, self.trust, self.readiness = (
            self._clip(self.distress), self._clip(self.trust), self._clip(self.readiness))
        # readiness grows with trust (persistent, delayed effect)
        self.readiness = self._clip(self.readiness + 0.03 * self.trust)

        # unattended known risk costs every step
        if self.risk and self.safety_checked and not terminated:
            reward -= 0.4

        # --- privacy -----------------------------------------------------------
        if self.shared_device and not self.privacy_on and not terminated:
            self.exposure = min(1.0, self.exposure + 0.06)
            if r.random() < 0.04 + 0.15 * self.exposure:
                reward -= 8.0
                terminated = True
                self.outcome = "PRIVACY BREACH: someone saw the screen"
                ev = self.outcome

        # --- drop-out ----------------------------------------------------------
        if not terminated and (self.trust < 0.05 or self.distress > 0.98):
            reward -= 5.0
            terminated = True
            self.outcome = "She left the session (drop-out)"
            ev = self.outcome

        truncated = (not terminated) and self.t >= self.max_steps
        # v2: danger missed. Ending (or timing out) on an at-risk survivor without a human hand-off.
        missed = not self.outcome.startswith(("At-risk", "Escalated", "She left", "PRIVACY"))
        if self.reward_version >= 2 and self.risk and (terminated or truncated) and missed:
            reward -= 8.0
            ev += " | DANGER MISSED"
        if truncated:
            self.outcome = "Session timed out"
        self.last_event = ev
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------ render ---
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import SafePathRenderer
            self._renderer = SafePathRenderer(offscreen=self.render_mode == "rgb_array")
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
