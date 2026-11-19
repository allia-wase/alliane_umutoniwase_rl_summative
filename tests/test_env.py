import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import SafePathEnv, N_ACTIONS, OBS_DIM


def test_gym_api():
    check_env(SafePathEnv(), skip_render_check=True)


def test_random_rollouts_terminate_and_stay_in_bounds():
    env = SafePathEnv()
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        assert obs.shape == (OBS_DIM,)
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            done = term or trunc
        assert env.t <= env.max_steps


def test_referral_without_consent_is_penalised():
    env = SafePathEnv()
    env.reset(seed=0)
    _, r, *_ = env.step(7)
    assert r < 0 and env.trust < 0.5


def test_action_effects_persist():
    env = SafePathEnv()
    env.reset(seed=1)
    t0 = env.trust
    for _ in range(3):
        env.step(0)
    assert env.trust > t0
    assert N_ACTIONS == 10
