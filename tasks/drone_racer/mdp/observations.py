# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""MDP observation functions for drone racing.

Provides observation terms that extract drone state information in various
reference frames (world, body, gate-relative) for the RL policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the body frame.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Linear velocity ``[vx, vy, vz]`` in body frame. Shape: ``(num_envs, 3)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    log(env, ["vx", "vy", "vz"], lin_vel)
    return lin_vel


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the body frame.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Angular velocity ``[wx, wy, wz]`` in body frame. Shape: ``(num_envs, 3)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    log(env, ["wx", "wy", "wz"], ang_vel)
    return ang_vel


def root_quat_w(
    env: ManagerBasedRLEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root orientation quaternion in the world frame.

    Args:
        env: The RL environment instance.
        make_quat_unique: If True, ensures the quaternion has a non-negative
            real part (w >= 0) to remove the double-cover ambiguity.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Quaternion ``[w, x, y, z]`` in world frame. Shape: ``(num_envs, 4)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    log(env, ["qw", "qx", "qy", "qz"], quat)
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_rotmat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root orientation as a flattened 3x3 rotation matrix in the world frame.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Row-major flattened rotation matrix. Shape: ``(num_envs, 9)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return flat_rotmat


def root_rotmat6d_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root orientation as a 6D rotation representation in the world frame.

    Uses the first two columns of the rotation matrix as the continuous
    6D representation.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        6D rotation representation. Shape: ``(num_envs, 6)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    rotmat6d = rotmat[:, :2, :].reshape(-1, 6)
    # logging purposes
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return rotmat6d


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root position in the world frame.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Position ``[x, y, z]`` in world frame. Shape: ``(num_envs, 3)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w
    log(env, ["px", "py", "pz"], position)
    return position


def root_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Drone pose expressed in the current target gate's reference frame.

    Transforms the drone's world-frame pose into the gate-local coordinate
    system by inverting the gate quaternion and rotating/translating accordingly.

    Args:
        env: The RL environment instance.
        command_name: Name of the command term providing the target gate pose.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Pose ``[x, y, z, qw, qx, qy, qz]`` in gate frame.
        Shape: ``(num_envs, 7)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    drone_pose_w = asset.data.root_state_w[:, :7]  # (num_envs, 7)

    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    drone_pos_w = drone_pose_w[:, :3]
    drone_quat_w = drone_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = drone_pos_w - gate_pos_w
    drone_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    drone_quat_g = math_utils.quat_mul(gate_quat_w_inv, drone_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([drone_pos_g, drone_quat_g], dim=-1)

    return position


def next_gate_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Next gate pose expressed in the current target gate's reference frame.

    Useful for providing the policy with look-ahead information about
    the upcoming gate relative to the one it is currently approaching.

    Args:
        env: The RL environment instance.
        command_name: Name of the command term providing gate poses.

    Returns:
        Pose ``[x, y, z, qw, qx, qy, qz]`` of the next gate in the
        current gate's frame. Shape: ``(num_envs, 7)``.
    """
    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    next_gate_pose_w = env.command_manager.get_term(command_name).next_gate  # (num_envs, 7)

    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    next_gate_pos_w = next_gate_pose_w[:, :3]
    next_gate_quat_w = next_gate_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = next_gate_pos_w - gate_pos_w
    next_gate_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    next_gate_quat_g = math_utils.quat_mul(gate_quat_w_inv, next_gate_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([next_gate_pos_g, next_gate_quat_g], dim=-1)

    return position


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target position expressed in the drone's body frame.

    Args:
        env: The RL environment instance.
        command_name: Name of the command term providing the target gate pose.
            Required when ``target_pos`` is ``None``.
        target_pos: Optional fixed target ``[x, y, z]`` in local env frame.
            When provided, overrides the command-based target.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Position vector to target in body frame. If command_name returns
        multiple gates ``(num_envs, n, 7)``, returns flattened relative
        positions ``(num_envs, 3*n)``. Otherwise ``(num_envs, 3)``.
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_poses = env.command_manager.get_term(command_name).command  # Shape: (num_envs, n, 7)

        # Check if we have multiple gates (3D tensor) or single gate (2D tensor)
        if len(target_poses.shape) == 3:
            # Multiple gates: (num_envs, n, 7)
            num_envs, num_gates, _ = target_poses.shape

            # Prepare robot poses for broadcasting
            robot_pos_w = asset.data.root_pos_w  # (num_envs, 3)
            robot_quat_w = asset.data.root_quat_w  # (num_envs, 4)

            # Initialize output tensor
            all_pos_b = torch.zeros(num_envs, num_gates, 3, device=asset.device)

            # Compute relative position for each gate
            for i in range(num_gates):
                gate_poses = target_poses[:, i, :]  # (num_envs, 7)
                pos_b, _ = math_utils.subtract_frame_transforms(
                    robot_pos_w, robot_quat_w, gate_poses[:, :3], gate_poses[:, 3:7]
                )
                all_pos_b[:, i, :] = pos_b

            # Flatten to (num_envs, 3*num_gates)
            pos_b = all_pos_b.view(num_envs, -1)
        else:
            # Single gate: (num_envs, 7)
            pos_b, _ = math_utils.subtract_frame_transforms(
                asset.data.root_pos_w, asset.data.root_quat_w, target_poses[:, :3], target_poses[:, 3:7]
            )

    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )
        pos_b, _ = math_utils.subtract_frame_transforms(
            asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor
        )

    return pos_b


def pos_error_w(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position error in world frame.

    Args:
        env: The RL environment instance.
        command_name: Name of the command term providing the target gate pose.
        target_pos: Optional fixed target ``[x, y, z]`` in local env frame.
        asset_cfg: Scene entity configuration identifying the robot asset.

    Returns:
        Position error vector in world frame. Shape: ``(num_envs, 3)``.
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    pos_error_w = target_pos_tensor - asset.data.root_pos_w
    return pos_error_w
