# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""MDP termination conditions for drone racing.

Provides termination functions that end episodes when the drone exceeds
safety boundaries relative to its target gate position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def flyaway(
    env: ManagerBasedRLEnv,
    distance: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the drone is too far from its target position.

    Computes the L2 distance between the drone and the current target (either
    the active gate command or a fixed world position). Returns ``True`` for
    environments where the distance exceeds the threshold.

    Args:
        env: The RL environment instance.
        distance: Maximum allowed distance in meters before termination.
        command_name: Name of the command term providing the target gate pose.
            Required when ``target_pos`` is ``None``.
        target_pos: Optional fixed target ``[x, y, z]`` in local env frame.
            When provided, overrides the command-based target.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Boolean tensor of shape ``(num_envs,)``. ``True`` where the episode
        should be terminated.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).immediate_target[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute distance
    distance_tensor = torch.linalg.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return distance_tensor > distance


def flip(env: ManagerBasedRLEnv, angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset roll or pitch exceeds the angle threshold.

    Args:
        env: The RL environment instance.
        angle: Maximum allowed roll/pitch angle in degrees.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Boolean tensor of shape ``(num_envs,)``. ``True`` where termination
        is triggered.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    current_angle = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    current_angle_wrapped_abs = [torch.abs(math_utils.wrap_to_pi(angle)) for angle in current_angle]
    threshold_rad = torch.tensor(angle * (torch.pi / 180.0), device=env.device)
    angle_exceeds_threshold = (current_angle_wrapped_abs[0] > threshold_rad) | (
        current_angle_wrapped_abs[1] > threshold_rad
    )
    return angle_exceeds_threshold
