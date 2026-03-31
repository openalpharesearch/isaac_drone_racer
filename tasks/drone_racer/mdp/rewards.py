# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""MDP reward functions for drone racing.

Provides reward terms based on position error, gate progress, gate passage,
heading alignment, and angular velocity penalties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared L2 position error between the drone and its target.

    Computes ``sum((pos - target)^2)`` per environment. Suitable as a
    quadratic penalty term in the reward function.

    Args:
        env: The RL environment instance.
        command_name: Name of the command term providing the target gate pose.
        target_pos: Optional fixed target ``[x, y, z]`` in local env frame.
            When provided, overrides the command-based target.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Scalar squared error per environment. Shape: ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute sum of squared errors
    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tanh-shaped position proximity reward.

    Returns ``1 - tanh(distance / std)``, yielding a value near 1.0 when the
    drone is close to the target and decaying smoothly toward 0.0 at larger
    distances. The ``std`` parameter controls the decay width.

    Args:
        env: The RL environment instance.
        std: Standard deviation controlling the tanh decay width in meters.
        command_name: Name of the command term providing the target gate pose.
            Required when ``target_pos`` is ``None``.
        target_pos: Optional fixed target ``[x, y, z]`` in local env frame.
            When provided, overrides the command-based target.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Proximity reward in [0, 1] per environment. Shape: ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Frame-to-frame progress toward the target gate.

    Positive when the drone moves closer to the gate between consecutive
    steps, negative when it moves away. Computed as the difference in
    L2 distance: ``prev_distance - current_distance``.

    Args:
        env: The RL environment instance.
        command_name: Name of the command term providing the target gate pose.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Signed distance delta per environment (m). Shape: ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).immediate_target[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    progress = prev_distance - current_distance

    return progress


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
) -> torch.Tensor:
    """Discrete reward/penalty for gate passage events.

    Returns +1 when the drone passes through the gate correctly, -1 when
    it crosses the gate plane but misses the opening, and 0 otherwise.

    Args:
        env: The RL environment instance.
        command_name: Name of the command term tracking gate passage state.

    Returns:
        Reward signal in {-1, 0, +1} per environment. Shape: ``(num_envs,)``.
    """
    missed = (-1.0) * env.command_manager.get_term(command_name).gate_missed
    passed = (1.0) * env.command_manager.get_term(command_name).gate_passed
    return missed + passed


def lookat_next_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential heading alignment reward toward the next gate.

    Computes the angle between the drone's body x-axis (forward direction)
    and the vector pointing to the next gate, then returns
    ``exp(-angle / std)`` so that perfect alignment yields 1.0.

    Args:
        env: The RL environment instance.
        std: Decay rate (radians) for the exponential shaping.
        command_name: Name of the command term providing the target gate pose.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Heading reward in (0, 1] per environment. Shape: ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = next_gate_pos - drone_pos
    vec_to_gate = math_utils.normalize(vec_to_gate)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    drone_x_axis = math_utils.quat_apply(drone_att, x_axis)
    drone_x_axis = math_utils.normalize(drone_x_axis)

    dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.exp(-angle / std)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared L2 penalty on body-frame angular velocity.

    Penalises excessive rotation rates to encourage smoother flight.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Sum of squared angular velocity components per env. Shape: ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
