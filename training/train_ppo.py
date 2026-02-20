"""
training/train_ppo.py
──────────────────────
Trains the PPO reinforcement learning agent for adaptive nutrient dosing.

Environment:  HydroponicEnv (custom OpenAI Gym)
Algorithm:    Proximal Policy Optimization (Stable-Baselines3)
State space:  [ec_current, ph_current, stage_norm, day_norm, hour_sin, hour_cos]
Action space: Discrete(3)  — 0=decrease, 1=maintain, 2=increase EC
Reward:       -|ec_current - ec_target| - 0.5 * penalty_for_extreme_action

Output saved to:  saved_models/ppo_dosing_policy.zip

Run:
  python training/train_ppo.py [--timesteps 200000] [--envs 4]
"""

import os
import argparse
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

os.makedirs("saved_models",    exist_ok=True)
os.makedirs("training/outputs", exist_ok=True)

GROWTH_STAGES = ["seedling", "vegetative", "flowering", "harvest"]
STAGE_DAYS    = [0, 14, 42, 72]
EC_TARGETS    = [0.8, 1.4, 1.8, 1.6]


def _stage_at_day(day: int) -> int:
    for i in reversed(range(len(STAGE_DAYS))):
        if day >= STAGE_DAYS[i]:
            return i
    return 0


# ─── Custom Gym Environment ───────────────────────────────────────────────────

class HydroponicEnv(gym.Env):
    """
    Single-tray hydroponic nutrient control environment.

    Observation: [ec, ph, stage_norm, day_norm, hour_sin, hour_cos]
    Action:      Discrete(3)  0=decrease 1=maintain 2=increase
    Episode:     One full 90-day crop cycle (8640 steps × 15min)
    """

    metadata = {"render_modes": []}

    def __init__(self, crop_type: str = "lettuce", noise_scale: float = 0.05):
        super().__init__()
        self.crop_type   = crop_type
        self.noise_scale = noise_scale
        self.rng         = np.random.default_rng()

        self.observation_space = spaces.Box(
            low   = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype="float32"),
            high  = np.array([5.0, 14.0, 1.0, 1.0, 1.0, 1.0],  dtype="float32"),
        )
        self.action_space = spaces.Discrete(3)   # 0=decrease, 1=maintain, 2=increase

        self._max_steps = 8640   # 90 days × 96 steps/day
        self._reset_state()

    def _reset_state(self):
        self._step     = 0
        self._day      = 0
        self._hour     = 6.0
        self._ec       = 1.0 + self.rng.uniform(-0.1, 0.1)
        self._ph       = 6.0 + self.rng.uniform(-0.2, 0.2)

    def _get_obs(self):
        stage    = _stage_at_day(self._day)
        day_norm = self._day / 90.0
        h        = self._hour
        return np.array([
            self._ec,
            self._ph,
            stage / 3.0,
            day_norm,
            math.sin(2 * math.pi * h / 24),
            math.cos(2 * math.pi * h / 24),
        ], dtype="float32")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        stage_i   = _stage_at_day(self._day)
        ec_target = EC_TARGETS[stage_i]

        # ── Apply dosing action ──
        ec_delta = {0: -0.1, 1: 0.0, 2: 0.1}[int(action)]
        self._ec += ec_delta

        # ── Plant uptake + noise ──
        self._ec -= self.rng.uniform(0.001, 0.004)
        self._ec += self.rng.normal(0, self.noise_scale * 0.05)
        self._ec  = float(np.clip(self._ec, 0.1, 4.5))

        # ── pH drift ──
        self._ph += 0.002 * math.sin(2 * math.pi * self._hour / 24)
        self._ph += self.rng.normal(0, 0.01)
        self._ph  = float(np.clip(self._ph, 4.0, 9.0))

        # ── Time advancement ──
        self._step  += 1
        self._hour   = (self._hour + 0.25) % 24   # 15-min tick
        self._day    = self._step // 96

        # ── Reward ──
        ec_error  = abs(self._ec - ec_target)
        # Penalise extreme actions (unnecessary dosing wastes nutrients)
        action_penalty = 0.0 if action == 1 else 0.3

        reward = -(ec_error + action_penalty)

        # Bonus for staying within ±0.1 of target
        if ec_error < 0.1:
            reward += 0.5

        terminated = self._step >= self._max_steps
        truncated  = False

        return self._get_obs(), float(reward), terminated, truncated, {}

    def render(self):
        stage = GROWTH_STAGES[_stage_at_day(self._day)]
        print(f"Day {self._day:3d} | EC {self._ec:.3f} | pH {self._ph:.3f} | Stage: {stage}")


# ─── Training ─────────────────────────────────────────────────────────────────

def train(timesteps: int = 200_000, n_envs: int = 4):
    print("=" * 60)
    print("  SmartHydro — PPO Nutrient Dosing RL Trainer")
    print("=" * 60)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, StopTrainingOnRewardThreshold
    )
    from stable_baselines3.common.monitor import Monitor
    import matplotlib.pyplot as plt

    # ── 1. Create vectorised environments ────────────────────────
    print(f"\n[1/4] Creating {n_envs} parallel environments…")
    env     = make_vec_env(HydroponicEnv, n_envs=n_envs)
    eval_env = Monitor(HydroponicEnv())

    # ── 2. Instantiate PPO ────────────────────────────────────────
    print("[2/4] Initialising PPO agent…")
    model = PPO(
        policy             = "MlpPolicy",
        env                = env,
        learning_rate      = 3e-4,
        n_steps            = 2048,
        batch_size         = 64,
        n_epochs           = 10,
        gamma              = 0.99,
        gae_lambda         = 0.95,
        clip_range         = 0.2,
        ent_coef           = 0.01,
        vf_coef            = 0.5,
        max_grad_norm      = 0.5,
        policy_kwargs      = dict(net_arch=[dict(pi=[128, 64], vf=[128, 64])]),
        verbose            = 1,
        tensorboard_log    = "training/outputs/ppo_tensorboard/",
    )
    print(f"      Policy network: {model.policy}")

    # ── 3. Train ──────────────────────────────────────────────────
    print(f"\n[3/4] Training for {timesteps:,} timesteps…")

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-0.15, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = "saved_models/",
        log_path             = "training/outputs/ppo_eval/",
        eval_freq            = max(5000 // n_envs, 1),
        callback_on_new_best = stop_callback,
        verbose              = 1,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    # ── 4. Save & Evaluate ────────────────────────────────────────
    print("\n[4/4] Saving policy and evaluating…")
    model.save("saved_models/ppo_dosing_policy")

    # Quick deterministic rollout
    obs, _ = eval_env.reset()
    total_reward = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = eval_env.step(action)
        total_reward += reward
        if done:
            break

    print(f"\n  Eval rollout (500 steps) cumulative reward: {total_reward:.2f}")
    print("  Policy saved → saved_models/ppo_dosing_policy.zip")

    # ── Reward curve ──────────────────────────────────────────────
    try:
        import pandas as pd
        log = pd.read_csv("training/outputs/ppo_eval/evaluations.npz",
                          allow_pickle=True)
    except Exception:
        pass  # tensorboard logs will have the curve

    print("\nDone ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO nutrient dosing agent")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total environment steps for training")
    parser.add_argument("--envs",      type=int, default=4,
                        help="Number of parallel environments")
    args = parser.parse_args()
    train(timesteps=args.timesteps, n_envs=args.envs)
