# Copyright (c) 2025, Kousheek Chakraborty
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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="The RL algorithm used for training the skrl agent.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--renderer",
    type=str,
    default="RayTracedLighting",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer to use.",
)
parser.add_argument("--log", type=int, default=None, help="Log the observations and metrics.")

# append AppLauncher cli args
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
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, Twist, Vector3

# ==================== OBSERVATION LAYOUT (edit to match your task) ====================
# Whiteboard flow: a 16-len tensor from env -> ROS
# Default mapping assumes:
#   0:4   -> quaternion [w,x,y,z]
#   4:7   -> position [px,py,pz]
#   7:10  -> linear velocity [vx,vy,vz]
#   10:13 -> angular velocity [wx,wy,wz]
#   13:16 -> body-frame goal delta [dx_b,dy_b,dz_b]  (or your last 3 entries)
OBS_MAP = {
    "quat":  (0, 4),
    "pos":   (4, 7),
    "lin_v": (7, 10),
    "ang_v": (10, 13),
    "goal_b":(13, 16),
}
assert sum((e - s) for s, e in OBS_MAP.values()) == 16, "OBS_MAP slices must total 16 elements."

# ==================== RC→ACTION MIXING (edit signs/gains to match your props/axes) ====================
# RC input ordering: [roll, pitch, thrust, yaw_rate]
# Action/motor order (per your whiteboard):
#   actions[0] = front-right (FR)
#   actions[1] = rear-left   (RL)
#   actions[2] = front-left  (FL)
#   actions[3] = rear-right  (RR)
#
# Allocation matrix A maps [roll, pitch, thrust, yaw] -> [FR, RL, FL, RR]
A_DEFAULT = np.array([
    #   roll   pitch  thrust  yaw
    [  -1.0,  +1.0,   1.0,   -1.0],  # FR
    [  -1.0,  -1.0,   1.0,   +1.0],  # RL
    [  +1.0,  +1.0,   1.0,   +1.0],  # FL
    [  +1.0,  -1.0,   1.0,   -1.0],  # RR
], dtype=np.float32)

# Per-channel RC gains (roll, pitch, thrust, yaw)
RC_GAINS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# Clamp range for env actions (change if your env expects 0..1)
ACTION_MIN = -1.0
ACTION_MAX =  1.0

def rc_to_actions(rc_vec, A=A_DEFAULT, rc_gains=RC_GAINS,
                  a_min=ACTION_MIN, a_max=ACTION_MAX):
    """
    rc_vec: [roll, pitch, thrust, yaw_rate]
    returns actions: [FR, RL, FL, RR]
    """
    u = rc_vec * rc_gains
    motors = A @ u  # shape (4,)
    motors = np.clip(motors, a_min, a_max)
    return motors

# -------------------- Simple buffer for latest RC command --------------------
class RCBuffer:
    def __init__(self):
        self._rc = np.zeros(4, dtype=np.float32)
        self._has_cmd = False

    def update(self, arr):
        if arr is None or len(arr) < 4:
            return
        self._rc = np.array(arr[:4], dtype=np.float32)
        self._has_cmd = True

    def get(self):
        return self._rc.copy(), self._has_cmd

# -------------------- ROS 2 node that bridges state pub + RC sub --------------------
class IsaacRosBridge(Node):
    def __init__(self, rc_buffer: RCBuffer):
        super().__init__("isaac_ros_bridge")
        self.rc_buffer = rc_buffer

        # State publishers
        self.state_pose_pub = self.create_publisher(Pose, "/drone/state/pose", 10)
        self.state_twist_pub = self.create_publisher(Twist, "/drone/state/twist", 10)
        self.state_goal_pub  = self.create_publisher(Vector3, "/drone/state/goal_b", 10)
        self.state_raw_pub   = self.create_publisher(Float32MultiArray, "/drone/state/raw", 10)

        # RC override subscriber (Float32MultiArray: [roll, pitch, thrust, yaw_rate])
        self.rc_sub = self.create_subscription(
            Float32MultiArray, "/rc/override", self._on_rc, 10
        )

    def _on_rc(self, msg: Float32MultiArray):
        self.rc_buffer.update(msg.data)

    def publish_state(self, obs_np: np.ndarray):
        # Raw obs (len=16)
        self.state_raw_pub.publish(Float32MultiArray(data=obs_np.astype(np.float32).tolist()))

        # Structured messages
        s, e = OBS_MAP["quat"];   quat = obs_np[s:e]
        s, e = OBS_MAP["pos"];    pos  = obs_np[s:e]
        s, e = OBS_MAP["lin_v"];  linv = obs_np[s:e]
        s, e = OBS_MAP["ang_v"];  angv = obs_np[s:e]
        s, e = OBS_MAP["goal_b"]; goal = obs_np[s:e]

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        # expecting [w,x,y,z]
        pose.orientation.w = float(quat[0])
        pose.orientation.x = float(quat[1])
        pose.orientation.y = float(quat[2])
        pose.orientation.z = float(quat[3])
        self.state_pose_pub.publish(pose)

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = float(linv[0]), float(linv[1]), float(linv[2])
        twist.angular.x, twist.angular.y, twist.angular.z = float(angv[0]), float(angv[1]), float(angv[2])
        self.state_twist_pub.publish(twist)

        goal_b = Vector3()
        goal_b.x, goal_b.y, goal_b.z = float(goal[0]), float(goal[1]), float(goal[2])
        self.state_goal_pub.publish(goal_b)

# ----------------------------------------------------------------------

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
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
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

import tasks  # noqa: F401
from utils.logger import CSVLogger

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent, bridged to ROS 2 for state pub + RC override actions."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.log and args_cli.num_envs > 1:
        raise ValueError("Logging is only supported for a single agent. Set --num_envs to 1.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.log:
        logger = CSVLogger(log_dir)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # -------------------- ROS 2 init + bridge --------------------
    rclpy.init(args=None)
    rc_buffer = RCBuffer()
    ros_node = IsaacRosBridge(rc_buffer)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    num_episode = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # Allow ROS callbacks (RC updates) to run
        executor.spin_once(timeout_sec=0.0)

        # --- Publish state to ROS ---
        # Convert obs to a flat numpy vector (len=16) for publishing
        if isinstance(obs, dict):
            # multi-agent dict -> take first agent's obs for publishing (adjust as needed)
            first_key = next(iter(obs))
            obs_tensor = obs[first_key]
        else:
            obs_tensor = obs

        # Ensure we have a 1D tensor (B=1) -> (16,)
        obs_np = obs_tensor.detach().cpu().numpy().reshape(-1)
        # Safety: only publish the first 16 entries even if wrapper adds extras
        if obs_np.shape[0] < 16:
            # pad if needed (unlikely in your setup)
            obs_np = np.pad(obs_np, (0, 16 - obs_np.shape[0]), mode="constant")
        else:
            obs_np = obs_np[:16]
        ros_node.publish_state(obs_np)

        # --- Build actions ---
        # Prefer RC override if any command has been received; otherwise use RL policy mean action
        rc_vec, has_cmd = rc_buffer.get()  # [roll, pitch, thrust, yaw_rate]
        if has_cmd:
            actions_np = rc_to_actions(rc_vec)  # [FR, RL, FL, RR]
            # shape to (num_envs, action_dim); here assume single env
            action_tensor = torch.from_numpy(actions_np).to(obs_tensor.device).unsqueeze(0)  # (1,4)
            actions = action_tensor
            if hasattr(env, "possible_agents"):
                # broadcast same RC action to all agents if MARL
                actions = {a: action_tensor for a in env.possible_agents}
        else:
            # Fallback to policy action (deterministic mean)
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])

        # --- Step env ---
        with torch.inference_mode():
            obs, rew, terminated, truncated, info = env.step(actions)

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if args_cli.log:
            # Handle bool or vectorized done flags
            term = (terminated is True) or (hasattr(terminated, "any") and terminated.any())
            trunc = (truncated is True) or (hasattr(truncated, "any") and truncated.any())
            if term or trunc:
                num_episode += 1
                logger.save()
                if num_episode >= args_cli.log:
                    break
            if isinstance(info, dict) and "metrics" in info:
                logger.log(info["metrics"])

    # close the simulator
    env.close()

    # shutdown ROS 2
    rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
