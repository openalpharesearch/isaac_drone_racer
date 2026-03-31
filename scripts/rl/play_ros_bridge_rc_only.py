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
from geometry_msgs.msg import Quaternion, Vector3


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
parser.add_argument("--save_cam", action = "store_true", default = False, help = "Save drone pov camera data")
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
from sensor_msgs.msg import Image as Ros_Image
from sensor_msgs.msg import Imu
from PIL import Image as PIL_Image
from cv_bridge import CvBridge # Import the key library

from geometry_msgs.msg import PoseStamped, TwistStamped
from autonomy_msgs.msg import DroneState  # ← your custom message
from mavros_msgs.msg import OverrideRCIn     # ← using OverrideRCIn

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# Initialize the CvBridge object
bridge = CvBridge()

# ==================== OBSERVATION LAYOUT (edit to match your task) ====================
# Whiteboard flow: a 16-len tensor from env -> ROS
# Default mapping assumes:
#   0:4   -> quaternion [w,x,y,z]
#   4:7   -> position [px,py,pz]
#   7:10  -> linear velocity [vx,vy,vz]
#   10:13 -> angular velocity [wx,wy,wz]
#   13:16 -> body-frame goal delta [dx_b,dy_b,dz_b]  (or your last 3 entries)
OBS_MAP = {
    "pos":  (0, 3),
    "quat":   (3, 7),
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

# RC channel mapping constants
RC_MID = 1500
RC_RANGE = 500  # ±500 from mid = 1000-2000
MAX_THRUST_N = 50.0  # Max thrust in Newtons


def rc_channels_to_motor_actions(channels: list) -> np.ndarray:
    """
    Convert RC channels to motor actions [-1, 1].
    
    RC channels (from autonomous_control_node):
        [0] Roll:     1000-2000 (1500 = neutral)
        [1] Pitch:    1000-2000 (1500 = neutral)
        [2] Throttle: 1000-2000 (1000 = min, 2000 = max)
        [3] Yaw:      1000-2000 (1500 = neutral)
    
    Motor output order: [FR, RL, FL, RR]
    """
    '''

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
    '''
    # Normalize channels to [-1, 1] (throttle to [0, 1] then to [-1, 1])
    roll = (channels[0] - RC_MID) / RC_RANGE
    pitch = (channels[1] - RC_MID) / RC_RANGE
    thrust = (channels[2] - 1000) / 1000  # 0-1 range
    thrust = thrust * 2 - 1  # Convert to [-1, 1]
    yaw = (channels[3] - RC_MID) / RC_RANGE
    
    # Clamp to valid range
    roll = np.clip(roll, -1, 1)
    pitch = np.clip(pitch, -1, 1)
    thrust = np.clip(thrust, -1, 1)
    yaw = np.clip(yaw, -1, 1)
    
    # Motor mixing matrix: [roll, pitch, thrust, yaw] -> [FR, RL, FL, RR]
    # Standard quadcopter "X" configuration
    A = np.array([
        [-1.0, +1.0, 1.0, -1.0],  # FR
        [-1.0, -1.0, 1.0, +1.0],  # RL
        [+1.0, +1.0, 1.0, +1.0],  # FL
        [+1.0, -1.0, 1.0, -1.0],  # RR
    ], dtype=np.float32)
    
    u = np.array([roll, pitch, thrust, yaw], dtype=np.float32)
    motors = A @ u
    motors = np.clip(motors, -1, 1)
    
    return motors


# ==================== NEW: Attitude Control + Mixer ====================

# PD gains for roll/pitch attitude regulation (tune as needed)
KP_ATT = np.array([3.0, 3.0])   # proportional gains for roll, pitch
KD_ATT = np.array([0.8, 0.8])   # derivative gains for roll, pitch

def extract_roll_pitch_from_quat(q):
    """
    q: quaternion [w, x, y, z]
    Returns roll, pitch (in radians)
    """
    w, x, y, z = q
    # roll (x-axis rotation)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    # pitch (y-axis rotation)
    pitch = np.arcsin(2*(w*y - z*x))
    return roll, pitch


def control_from_obs_and_channels(obs_np, channels):
    """
    NEW FUNCTION:
        Inputs:
            obs_np   - 1D observation array (len >= 16)
            channels - RC channels [roll, pitch, throttle, yaw_rate]

        Output:
            motor commands [FR, RL, FL, RR] in [-1, 1]
    """

    # ============================
    # 1. Parse observation
    # ============================
    quat = obs_np[3:7]                       # [w,x,y,z]
    ang_vel_b = obs_np[10:13]                # [wx,wy,wz] body rates

    roll_now, pitch_now = extract_roll_pitch_from_quat(quat)

    wx, wy, wz = ang_vel_b

    # ============================
    # 2. Desired GLOBAL roll/pitch
    # ============================
    roll_des  = np.clip((channels[0] - RC_MID) / RC_RANGE, -1.0, 1.0)
    pitch_des = np.clip((channels[1] - RC_MID) / RC_RANGE, -1.0, 1.0)

    # scale to radians (±1 → ±30°, tune as needed)
    MAX_ANGLE_RAD = np.deg2rad(30.0)
    roll_des_rad  = roll_des  * MAX_ANGLE_RAD
    pitch_des_rad = pitch_des * MAX_ANGLE_RAD

    # ============================
    # 3. Error terms
    # ============================
    e_roll  = roll_des_rad  - roll_now
    e_pitch = pitch_des_rad - pitch_now

    # body angular rates act as derivative feedback
    ed_roll  = -wx
    ed_pitch = -wy

    # ============================
    # 4. PD attitude controller
    # ============================
    roll_cmd  = KP_ATT[0] * e_roll  + KD_ATT[0] * ed_roll
    pitch_cmd = KP_ATT[1] * e_pitch + KD_ATT[1] * ed_pitch

    # normalize to [-1,1]
    roll_cmd  = np.clip(roll_cmd,  -1.0, 1.0)
    pitch_cmd = np.clip(pitch_cmd, -1.0, 1.0)

    # ============================
    # 5. Thrust + yaw remain LOCAL
    # ============================
    thrust_raw = np.clip((channels[2] - 1000) / 1000.0, 0.0, 1.0)
    thrust_centered = thrust_raw * 2.0 - 1.0

    yaw_cmd = np.clip((channels[3] - RC_MID) / RC_RANGE, -1.0, 1.0)

    # ============================
    # 6. Mixed command vector
    # ============================
    u = np.array([roll_cmd, pitch_cmd, thrust_centered, yaw_cmd], dtype=np.float32)

    motors = A_DEFAULT @ u
    motors = np.clip(motors, -1.0, 1.0)

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
        # consume the command so it can only run once
        if self._has_cmd:
            out = self._rc.copy()
            self._has_cmd = False
            self._rc = np.zeros(4, dtype=np.float32)
            return out, True
        else:
            return np.zeros(4, dtype=np.float32), False
# -------------------- ROS 2 node that bridges state pub + RC sub --------------------

SENSOR_QOS = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST
)

CAM_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST
)

# For critical command and control data - uses RELIABLE
COMMAND_QOS = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST
)


class IsaacRosBridge(Node):
    def __init__(self, rc_buffer: RCBuffer):
        super().__init__("isaac_ros_bridge")
        self.rc_buffer = rc_buffer

        # State publishers
        self.state_pose_pub = self.create_publisher(Pose, "/drone/state/pose", 10)
        self.state_twist_pub = self.create_publisher(Twist, "/drone/state/twist", 10)
        self.state_goal_pub  = self.create_publisher(Vector3, "/drone/state/goal_b", 10)
        self.state_raw_pub   = self.create_publisher(Float32MultiArray, "/drone/state/raw", 10)

        # compatible with main flightstack cam
        self.image_pub = self.create_publisher(Ros_Image, "/camera/image_raw", qos_profile=CAM_QOS)

        # publish the full drone state to be compatible with main flightstack
        self.drone_state_pub = self.create_publisher(DroneState, "/drone/state", qos_profile=SENSOR_QOS)

        self.imu_quat_pub = self.create_publisher(Quaternion, "/drone/imu/orientation", 10)
        self.imu_ang_pub  = self.create_publisher(Vector3, "/drone/imu/angular_velocity",10)
        self.imu_acc_pub  = self.create_publisher(Vector3, "/drone/imu/linear_acceleration",10)

        # publish full imu state for imu main flightstack as well
        self.imu_pub = self.create_publisher(Imu, "/drone/imu/data", qos_profile=SENSOR_QOS)


        # RC override subscriber (Float32MultiArray: [roll, pitch, thrust, yaw_rate])
        self.rc_sub = self.create_subscription(
            OverrideRCIn, "/rc/override", self._on_rc, qos_profile=COMMAND_QOS
        )

    def _on_rc(self, msg: OverrideRCIn):
        self.rc_buffer.update(msg.channels)
    def publish_cam(self, img: np.ndarray, frame_id: int):
        img_uint8 = img.astype(np.uint8) 
    
        # Convert NumPy array to ROS Image message
        #    'rgb8' specifies 8-bit R, G, B channels
        try:
            ros_image_msg = bridge.cv2_to_imgmsg(img_uint8, encoding="rgb8")
        except Exception as e:
            return

        ros_image_msg.header.frame_id = str(frame_id) # desired frame
        # also need timestamp
        ros_image_msg.header.stamp = self.get_clock().now().to_msg()
        
        # 4. Publish the message
        self.image_pub.publish(ros_image_msg)

    def publish_imu(self, imu_data):
        quat_msg = Quaternion()
        ang_vel_msg = Vector3()
        lin_acc_msg = Vector3()
        # Orientation (world frame quaternion)
        quat_msg.w = float(imu_data.quat_w[0][0])
        quat_msg.x = float(imu_data.quat_w[0][1])
        quat_msg.y = float(imu_data.quat_w[0][2])
        quat_msg.z = float(imu_data.quat_w[0][3])

        # Angular velocity (body frame)
        ang_vel_msg.x = float(imu_data.ang_vel_b[0][0])
        ang_vel_msg.y = float(imu_data.ang_vel_b[0][1])
        ang_vel_msg.z = float(imu_data.ang_vel_b[0][2])

        # Linear acceleration (body frame)
        lin_acc_msg.x = float(imu_data.lin_acc_b[0][0])
        lin_acc_msg.y = float(imu_data.lin_acc_b[0][1])
        lin_acc_msg.z = float(imu_data.lin_acc_b[0][2])

        # publish full imu message with timestamp
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = "imu_link"  # or your desired frame   
        imu_msg.orientation = quat_msg
        imu_msg.angular_velocity = ang_vel_msg
        imu_msg.linear_acceleration = lin_acc_msg
        self.imu_pub.publish(imu_msg)


        self.imu_quat_pub.publish(quat_msg)
        self.imu_ang_pub.publish(ang_vel_msg)
        self.imu_acc_pub.publish(lin_acc_msg)        


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

        # full drone state message as the result of localization
        drone_state = DroneState()
        drone_state.header.stamp = self.get_clock().now().to_msg()
        drone_state.header.frame_id = "drone_base_link"  # or your desired frame
        
        pose_stamped = PoseStamped()
        pose_stamped.header = drone_state.header
        pose_stamped.pose = pose
        drone_state.pose = pose_stamped

        twist_stamped = TwistStamped()
        twist_stamped.header = drone_state.header
        twist_stamped.twist = twist
        drone_state.twist = twist_stamped

        self.drone_state_pub.publish(drone_state)


# ----------------------------------------------------------------------

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

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

    # saving camera_output locally
    camera_out_dir = f"camerapov/{args_cli.task}"
    if args_cli.save_cam:
        os.makedirs(camera_out_dir, exist_ok = True)

    # simulate environment
    frame_count = 0
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
        # create an action tensor that is just 4 zeros if no RC command
        actions = torch.Tensor(4)
        if has_cmd:
            # actions_np = rc_channels_to_motor_actions(rc_vec)  # [FR, RL, FL, RR]
            actions_np = control_from_obs_and_channels(obs_np, rc_vec)
            # shape to (num_envs, action_dim); here assume single env
            action_tensor = torch.from_numpy(actions_np).to(obs_tensor.device).unsqueeze(0)  # (1,4)
            actions = action_tensor
            if hasattr(env, "possible_agents"):
                # broadcast same RC action to all agents if MARL
                actions = {a: action_tensor for a in env.possible_agents}
            print(actions)
        else:
            actions = torch.zeros(4)

        # --- Step env ---
        with torch.inference_mode():
            obs, rew, terminated, truncated, info = env.step(actions)

        camera_data = env.scene["tiled_camera"].data.output["rgb"].detach().cpu().numpy().squeeze()
        imu_data = env.scene["imu"].data


        if args_cli.save_cam:
            frame_filename = os.path.join(
                        camera_out_dir, 
                        f"frame_{frame_count}.png"
                    )
            PIL_Image.fromarray(camera_data).save(frame_filename)
        frame_count += 1
        ros_node.publish_cam(camera_data,frame_count)
        ros_node.publish_imu(imu_data)

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    # shutdown ROS 2
    rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
