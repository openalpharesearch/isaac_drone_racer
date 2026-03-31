# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""MDP command generators for gate-targeting drone racing.

Provides command terms that track gate progression and detect gate passage
events during autonomous drone racing episodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import cv2
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass

from .events import reset_after_prev_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GateTargetingCommand(CommandTerm):
    """Command generator that tracks gate progression along a racing track.

    Maintains a per-environment gate index and detects gate passage (or miss)
    by projecting the drone position onto the gate normal between consecutive
    steps. The command tensor is the world-frame pose of the next target gate(s).
    """

    cfg: GateTargetingCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: GateTargetingCommandCfg, env: ManagerBasedEnv):
        """Initialize the gate-targeting command generator.

        Args:
            cfg: Configuration specifying assets, gate geometry, and start mode.
            env: The simulation environment.
        """
        super().__init__(cfg, env)

        self.cfg = cfg

        # FPV video recording
        if self.cfg.record_fpv:
            self.video_id = 0
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera")
            self.sensor: TiledCamera = self._env.scene.sensors[self.sensor_cfg.name]

        # extract the robot and track for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.track: RigidObjectCollection = env.scene[cfg.track_name]
        self.gate_size = cfg.gate_size
        self.num_gates = self.track.num_objects

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in simulation world frame for next n gates
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.prev_robot_pos_w = self.robot.data.root_pos_w
        self._gate_missed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.next_gate_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.next_n_gates_w = torch.zeros(self.num_envs, self.cfg.n, 7, device=self.device)

    def __str__(self) -> str:
        """Return a human-readable summary of the command generator."""
        msg = "GateTargetingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tNumber of gates in command: {self.cfg.n}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose commands for the next n gates. Shape is (num_envs, n, 7).

        For each gate, the first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.next_n_gates_w

    @property
    def immediate_target(self) -> torch.Tensor:
        """The immediate target gate (first of the n gates). Shape is (num_envs, 7).

        This is the gate used for gate passing logic and immediate navigation.
        """
        return self.next_n_gates_w[:, 0, :]

    @property
    def gate_missed(self) -> torch.Tensor:
        """Boolean mask for environments where the drone missed the gate this step."""
        return self._gate_missed

    @property
    def gate_passed(self) -> torch.Tensor:
        """Boolean mask for environments where the drone passed the gate this step."""
        return self._gate_passed

    @property
    def previous_pos(self) -> torch.Tensor:
        """The drone's world position from the previous step. Shape: ``(num_envs, 3)``."""
        return self.prev_robot_pos_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update internal metrics (no-op for this command term)."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample gate targets and optionally reset drone pose.

        On resample, optionally saves the current FPV recording, resets the
        target gate index (either to 0, 1, or a random gate), and if
        ``randomise_start`` is enabled, repositions the drone behind the
        previous gate via :func:`.events.reset_after_prev_gate`.

        Args:
            env_ids: Environment indices being resampled.
        """
        # Release and reinitialize video writer only after the first iteration
        if hasattr(self, "out") and self.cfg.record_fpv:
            self.out.release()
            print(f"FPV video saved as fpv_{self.video_id}.mp4")
            self.video_id += 1

        if self.cfg.record_fpv:
            self.out = cv2.VideoWriter(f"fpv_{self.video_id}.mp4", self.fourcc, 100, (1000, 1000))

        if self.cfg.randomise_start is None:
            self.next_gate_idx[env_ids] = 0

        else:
            if self.cfg.randomise_start:
                self.next_gate_idx[env_ids] = torch.randint(
                    low=0, high=self.num_gates, size=(len(env_ids),), device=self.device, dtype=torch.int32
                )
            else:
                self.next_gate_idx[env_ids] = 1

            gate_indices = self.next_gate_idx - 1
            gate_positions = self.track.data.object_com_pos_w[self.env_ids, gate_indices]
            gate_orientations = self.track.data.object_quat_w[self.env_ids, gate_indices]
            gate_w = torch.cat([gate_positions, gate_orientations], dim=1)

            reset_after_prev_gate(
                env=self._env,
                env_ids=env_ids,
                gate_pose=gate_w,
                pose_range={
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-torch.pi / 4, torch.pi / 4),
                    "pitch": (-torch.pi / 4, torch.pi / 4),
                    "yaw": (-torch.pi / 4, torch.pi / 4),
                },
                velocity_range={
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                asset_cfg_name=self.cfg.asset_name,
            )

    def _update_command(self):
        """Update gate command and detect gate passage events.

        Refreshes the target gate pose from the track, then checks if the drone
        crossed the gate plane between the previous and current step by
        projecting positions onto the gate's forward normal. Classifies the
        crossing as a pass (within gate bounds) or miss (outside bounds) and
        advances the gate index on successful passage.
        """
        if self.cfg.record_fpv:
            image = self.sensor.data.output["rgb"][0].cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.out.write(image)

        # Update next n gates for all environments
        for i in range(self.cfg.n):
            gate_idx = (self.next_gate_idx + i) % self.num_gates
            gate_positions = self.track.data.object_com_pos_w[self.env_ids, gate_idx]
            gate_orientations = self.track.data.object_quat_w[self.env_ids, gate_idx]
            self.next_n_gates_w[:, i, :] = torch.cat([gate_positions, gate_orientations], dim=1)

        # Gate passing logic using only the first gate (index 0)
        first_gate_w = self.next_n_gates_w[:, 0, :]  # Shape: (num_envs, 7)
        (roll, pitch, yaw) = math_utils.euler_xyz_from_quat(first_gate_w[:, 3:7])
        normal = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        pos_old_projected = (self.prev_robot_pos_w[:, 0] - first_gate_w[:, 0]) * normal[:, 0] + (
            self.prev_robot_pos_w[:, 1] - first_gate_w[:, 1]
        ) * normal[:, 1]
        pos_new_projected = (self.robot.data.root_pos_w[:, 0] - first_gate_w[:, 0]) * normal[:, 0] + (
            self.robot.data.root_pos_w[:, 1] - first_gate_w[:, 1]
        ) * normal[:, 1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)

        self._gate_passed = passed_gate_plane & (
            torch.all(torch.abs(self.robot.data.root_pos_w - first_gate_w[:, :3]) < (self.gate_size / 2), dim=1)
        )

        self._gate_missed = passed_gate_plane & (
            torch.any(torch.abs(self.robot.data.root_pos_w - first_gate_w[:, :3]) > (self.gate_size / 2), dim=1)
        )

        # Update next gate target for the envs that passed the gate
        self.next_gate_idx[self._gate_passed] += 1
        self.next_gate_idx = self.next_gate_idx % self.num_gates

        self.prev_robot_pos_w = self.robot.data.root_pos_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Toggle debug visualisation markers for gate targets and drone pose.

        Args:
            debug_vis: Whether to enable or disable the debug markers.
        """
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                # -- goal pose
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                # -- current body pose
                self.drone_visualizer = VisualizationMarkers(self.cfg.drone_visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
            self.drone_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.drone_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback invoked each frame to update debug visualisation markers.

        Args:
            event: The visualisation callback event (unused).
        """
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers - visualize only the first gate
        first_gate_w = self.next_n_gates_w[:, 0, :]  # Shape: (num_envs, 7)
        self.target_visualizer.visualize(first_gate_w[:, :3], first_gate_w[:, 3:])
        self.drone_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)


@configclass
class GateTargetingCommandCfg(CommandTermCfg):
    """Configuration for :class:`GateTargetingCommand`.

    Specifies the robot and track assets, gate geometry, start randomisation
    behaviour, and debug visualisation marker settings.
    """

    class_type: type = GateTargetingCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    track_name: str = MISSING
    """Name of the gate track :class:`RigidObjectCollection` in the scene."""

    randomise_start: bool | None = None
    """Controls starting gate selection on reset.

    - ``None``: always start at gate 0.
    - ``False``: always start at gate 1.
    - ``True``: uniformly sample a random starting gate.
    """

    record_fpv: bool = False
    """If True, record FPV camera frames as an MP4 video each episode."""

    gate_size: float = 1.5
    """Full gate opening size in meters (axis-aligned bounding box half-width
    used to classify pass vs. miss)."""

    n: int = 5
    """Number of next gates to include in observation."""

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """Visualisation marker config for the target gate pose."""

    drone_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """Visualisation marker config for the drone body pose."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    target_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    drone_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
