# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""Diagnostic script for running the drone racer RL environment with zero actions.

Launches the IsaacSim application and steps the DroneRacerEnvCfg environment
while sending zero-valued actions. Useful for verifying environment setup,
observation pipelines, and reward signals without a trained policy.

Usage::

    python scripts/test.py --num_envs 16 [--headless]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test script to run RL environment directly.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from isaaclab.envs import ManagerBasedRLEnv

from tasks.drone_racer.drone_racer_env_cfg import DroneRacerEnvCfg


def main():
    """Run the environment loop with zero actions until the simulator exits.

    Creates a :class:`ManagerBasedRLEnv` from :class:`DroneRacerEnvCfg`,
    resets it periodically, and steps with zero actions so that the
    environment dynamics can be observed in the viewer.
    """
    env_cfg = DroneRacerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():

            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, :] = 0.0

            obs, rew, terminated, truncated, info = env.step(actions)
            # print(rew)
            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
