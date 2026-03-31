# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""MDP action space definitions for drone racing.

Defines body-rate control actions that map neural network outputs to thrust
and torque commands applied to the drone body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import BodyRateController
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ControlAction(ActionTerm):
    """Body-rate control action term for quadrotor drones.

    Maps neural network outputs (4-dimensional, normalized to [-1, 1]) to
    collective thrust and body-frame torques applied as external forces to
    the drone rigid body.
    """

    cfg: ControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the control action term.

        Args:
            cfg: Action term configuration specifying controller parameters.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._twr_default = torch.full((self.num_envs, 1), self.cfg.thrust_weight_ratio, device=self.device)
        self._twr = torch.full((self.num_envs, 1), self.cfg.thrust_weight_ratio, device=self.device)
        self._max_thrust = (
            self._robot.data.default_mass.sum(dim=1, keepdim=True)
            * -self._env.sim.cfg.gravity[-1]
            * self.cfg.thrust_weight_ratio
        ).to(self.device)

        self._rate_controller = BodyRateController(
            self.num_envs,
            self._robot.data.default_inertia[:, 0].view(-1, 3, 3),
            torch.eye(3) * self.cfg.k_rates,
            self.device,
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """The dimension of the action space.

        Returns:
            Number of action components (4: one per motor).
        """
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw (unprocessed) actions from the policy. Shape: ``(num_envs, 4)``."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions after mapping to thrust and torques.

        Returns:
            Tensor of ``[collective_thrust, torque_x, torque_y, torque_z]``
            per environment. Shape: ``(num_envs, 4)``.
        """
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether this action term has a debug visualization implementation."""
        return False

    @property
    def twr(self) -> torch.Tensor:
        """Thrust to weight ratio."""
        return self._twr

    @twr.setter
    def twr(self, value: torch.Tensor):
        """Set thrust to weight ratio."""
        self._twr = value

    @property
    def twr_default(self) -> torch.Tensor:
        """Thrust to weight ratio."""
        return self._twr_default

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Process raw policy actions into thrust and torque commands.

        Clamps actions to [-1, 1], maps thrust to [0, max_thrust], maps
        rates to angular velocity targets, and computes body torques via
        the rate controller.

        Args:
            actions: Raw actions from the policy. Shape: ``(num_envs, 4)``.
        """
        self._raw_actions[:] = actions.clone()
        clamped = self._raw_actions.clamp_(-1.0, 1.0)
        log(self._env, ["a1", "a2", "a3", "a4"], actions)
        log(self._env, ["a1_clamped", "a2_clamped", "a3_clamped", "a4_clamped"], clamped)

        # Clamp rates setpoint and total thrust
        # Calculate wrench based on rates setpoint
        # Calculate thrust setpoint based on wrench and allocation inverse
        # Clamp thrust setpoint

        # print(self._robot.data.default_mass.sum(dim=1, keepdim=True))
        # print(self._robot.data.default_inertia)

        mapped = clamped.clone()
        mapped[:, :1] = (mapped[:, :1] + 1) / 2
        mapped[:, :1] *= self._max_thrust
        mapped[:, 1:] *= torch.tensor(self.cfg.max_ang_vel, device=self.device, dtype=self._raw_actions.dtype)
        mapped[:, 1:] = self._rate_controller.compute_moment(mapped[:, 1:], self._robot.data.root_ang_vel_b)
        log(self._env, ["T", "rate1", "rate2", "rate3"], mapped)
        self._processed_actions = mapped

    def apply_actions(self):
        """Apply computed thrust and torques as external forces on the drone body.

        Sets the z-axis thrust and xyz torques on the robot body via the
        Isaac Sim external force/torque API.
        """
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:]
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        """Reset action buffers and robot joint state.

        Args:
            env_ids: Environment indices to reset. If ``None`` or all envs,
                resets every environment.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@configclass
class ControlActionCfg(ActionTermCfg):
    """Configuration for :class:`ControlAction`.

    Specifies controller parameters and rate limits for the quadrotor
    body-rate controller.
    """

    class_type: type[ActionTerm] = ControlAction

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    thrust_weight_ratio: float = 2.5
    """Thrust weight ratio of the drone."""

    max_ang_vel: list[float] = [3.5, 3.5, 3.5]
    """Maximum angular velocity in rad/s."""

    k_rates: float = 0.01
    """Proportional gain for angular velocity error."""
