# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="The ML framework used for training the skrl agent.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP","PPO","IPPO","MAPPO"], help="The RL algorithm used for training the skrl agent.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting","PathTracing"], help="Renderer to use.")
parser.add_argument("--log", type=int, default=None, help="Log the observations and metrics.")
parser.add_argument("--rc-timeout", type=float, default=0.25, help="RC command watchdog timeout (seconds).")

# append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# -------------------- ROS 2 / NumPy / msgs --------------------
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TwistStamped
from autonomy_msgs.msg import DroneState  # ← your custom message
from mavros_msgs.msg import OverrideRCIn     # ← using OverrideRCIn

# ==================== OBSERVATION LAYOUT (edit to match your task) ====================
# Default mapping assumes:
#   0:4   -> quaternion [w,x,y,z]
#   4:7   -> position [px,py,pz]
#   7:10  -> linear velocity [vx,vy,vz]
#   10:13 -> angular velocity [wx,wy,wz]
#   13:16 -> body-frame goal delta [dx_b,dy_b,dz_b]
OBS_MAP = {
    "quat":  (0, 4),
    "pos":   (4, 7),
    "lin_v": (7, 10),
    "ang_v": (10, 13),
    "goal_b":(13,16),
}
assert sum((e-s) for s,e in OBS_MAP.values()) == 16, "OBS_MAP slices must total 16 elements."

# ==================== RC→ACTION MIXING (edit signs/gains to match your props/axes) ====================
# RC input ordering: [roll, pitch, thrust, yaw_rate]
# Action/motor order (per your whiteboard):
#   actions[0] = front-right (FR)
#   actions[1] = rear-left   (RL)
#   actions[2] = front-left  (FL)
#   actions[3] = rear-right  (RR)
A_DEFAULT = np.array([
    # roll  pitch  thrust  yaw
    [ -1.0, +1.0,  1.0, -1.0],  # FR
    [ -1.0, -1.0,  1.0, +1.0],  # RL
    [ +1.0, +1.0,  1.0, +1.0],  # FL
    [ +1.0, -1.0,  1.0, -1.0],  # RR
], dtype=np.float32)
RC_GAINS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
ACTION_MIN = -1.0
ACTION_MAX =  1.0

def rc_to_actions(rc_vec, A=A_DEFAULT, rc_gains=RC_GAINS, a_min=ACTION_MIN, a_max=ACTION_MAX):
    """
    rc_vec: [roll, pitch, thrust, yaw_rate]
    returns actions: [FR, RL, FL, RR]
    """
    u = rc_vec * rc_gains
    motors = A @ u
    motors = np.clip(motors, a_min, a_max)
    return motors

# -------------------- Ephemeral RC buffer with timeout --------------------
class RCBuffer:
    """
    Consume-once RC commands with watchdog timeout.
    - Each received RC msg is applied for *one* step only, then cleared.
    - If no RC arrives for 'timeout_sec', it auto-clears to neutral zeros.
    """
    def __init__(self, timeout_sec: float = 0.25):
        self._rc      = np.zeros(4, dtype=np.float32)
        self._has_cmd = False
        self._last_t  = 0.0
        self._timeout = timeout_sec

    def update(self, arr):
        if arr is None or len(arr) < 4:
            return
        self._rc[:] = np.array(arr[:4], dtype=np.float32)
        self._has_cmd = True
        self._last_t = time.monotonic()

    def get(self, consume: bool = True):
        now = time.monotonic()
        if (not self._has_cmd) or ((now - self._last_t) > self._timeout):
            self._has_cmd = False
            self._rc[:] = 0.0
            return self._rc.copy(), False
        out = self._rc.copy()
        if consume:
            self._has_cmd = False
            self._rc[:] = 0.0
        return out, True

# -------------------- ROS 2 bridge node: state publishers + RC subscriber --------------------
class IsaacRosBridge(Node):
    def __init__(self, rc_buffer: RCBuffer, frame_map="map", frame_base="base_link"):
        super().__init__("isaac_ros_bridge")
        self.rc_buffer = rc_buffer
        self.frame_map  = frame_map
        self.frame_base = frame_base

        # State publisher (publishing DroneState)
        self.state_pub = self.create_publisher(DroneState, "/drone/state", 10)

        # RC override subscriber
        self.rc_sub = self.create_subscription(
            OverrideRCIn,
            "/mavros/rc/override",
            self._on_rc_override,
            10
        )

    def _on_rc_override(self, msg: OverrideRCIn):
        # Example extraction: use channels[0]=roll, [1]=pitch, [2]=thrust, [3]=yaw_rate
        ch = list(msg.channels)
        # Map raw channel PWM values (1000-2000) to ±1 range (example):
        mid   = 1500.0
        scale = 500.0
        roll     = (ch[0] - mid) / scale
        pitch    = (ch[1] - mid) / scale
        thrust   = (ch[2] - mid) / scale
        yaw_rate  = (ch[3] - mid) / scale
        rc_vec = [roll, pitch, thrust, yaw_rate]
        self.rc_buffer.update(rc_vec)

    def publish_state(self, obs_np: np.ndarray):
        # Ensure at least elements exist for mapping
        if obs_np.shape[0] < 13:
            obs_np = np.pad(obs_np, (0,13-obs_np.shape[0]), mode="constant")

        stamp = self.get_clock().now().to_msg()

        # Build DroneState message
        msg = DroneState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_map

        # PoseStamped
        s,e = OBS_MAP["quat"]; quat  = obs_np[s:e]
        s,e = OBS_MAP["pos"];   pos   = obs_np[s:e]

        msg.pose = PoseStamped()
        msg.pose.header.stamp = stamp
        msg.pose.header.frame_id = self.frame_map
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.w = float(quat[0])
        msg.pose.pose.orientation.x = float(quat[1])
        msg.pose.pose.orientation.y = float(quat[2])
        msg.pose.pose.orientation.z = float(quat[3])

        # TwistStamped
        s,e = OBS_MAP["lin_v"]; linv = obs_np[s:e]
        s,e = OBS_MAP["ang_v"]; angv = obs_np[s:e]

        msg.twist = TwistStamped()
        msg.twist.header.stamp = stamp
        msg.twist.header.frame_id = self.frame_base
        msg.twist.twist.linear.x  = float(linv[0])
        msg.twist.twist.linear.y  = float(linv[1])
        msg.twist.twist.linear.z  = float(linv[2])
        msg.twist.twist.angular.x = float(angv[0])
        msg.twist.twist.angular.y = float(angv[1])
        msg.twist.twist.angular.z = float(angv[2])

        # Publish
        self.state_pub.publish(msg)

# ----------------------------------------------------------------------

# Check skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install supported version using 'pip install skrl>={SKRL_VERSION}'")
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
import tasks  # noqa: F401
from utils.logger import CSVLogger

algorithm = args_cli.algorithm.lower()

def main():
    """Play with skrl agent, publish DroneState, accept OverrideRCIn commands (ephemeral)."""
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    if args_cli.log and args_cli.num_envs > 1:
        raise ValueError("Logging is only supported for a single agent. Set --num_envs to 1.")

    # Parse config / load env / checkpoint
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    log_root_path = os.path.join("logs","skrl",experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt
    if args_cli.log:
        logger = CSVLogger(log_dir)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir,"videos","play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"]     = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # ROS 2 init + bridge
    rclpy.init(args=None)
    rc_buffer = RCBuffer(timeout_sec=args_cli.rc_timeout)
    ros_node = IsaacRosBridge(rc_buffer)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)

    # Reset environment
    obs, _ = env.reset()
    timestep   = 0
    num_episode = 0

    # Simulation loop
    while simulation_app.is_running():
        start_time = time.time()
        executor.spin_once(timeout_sec=0.0)

        # Get obs vector
        if isinstance(obs, dict):
            first_key = next(iter(obs))
            obs_tensor = obs[first_key]
        else:
            obs_tensor = obs
        obs_np = obs_tensor.detach().cpu().numpy().reshape(-1)
        if obs_np.shape[0] < 16:
            obs_np = np.pad(obs_np, (0, 16-obs_np.shape[0]), mode="constant")
        else:
            obs_np = obs_np[:16]

        # Publish DroneState
        ros_node.publish_state(obs_np)

        # Build actions (no action unless RC present)
        rc_vec, has_cmd = rc_buffer.get(consume=True)
        if has_cmd:
            actions_np = rc_to_actions(rc_vec)
        else:
            actions_np = np.zeros(4, dtype=np.float32)

        action_tensor = torch.from_numpy(actions_np).to(obs_tensor.device).unsqueeze(0)
        actions = action_tensor
        if hasattr(env, "possible_agents"):
            actions = {a: action_tensor for a in env.possible_agents}

        with torch.inference_mode():
            obs, rew, terminated, truncated, info = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and (sleep_time > 0):
            time.sleep(sleep_time)

        if args_cli.log:
            term = (terminated is True) or (hasattr(terminated,"any") and terminated.any())
            trunc = (truncated is True) or (hasattr(truncated,"any") and truncated.any())
            if term or trunc:
                num_episode += 1
                logger.save()
                if num_episode >= args_cli.log:
                    break
            if isinstance(info, dict) and ("metrics" in info):
                logger.log(info["metrics"])

    # Close environment and ROS
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()
