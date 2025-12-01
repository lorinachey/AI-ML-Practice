#!/usr/bin/env python3
"""
Simple single robot replay script for IsaacLab policies with data logging.
This is a minimal script based on the play.py structure.

Usage:
    python replay_single_simple.py --checkpoint /path/to/policy.pt --num_steps 1000
"""
import argparse
from isaaclab.app import AppLauncher

DEFAULT_POLICY_PATH = "/home/lorin-cairo/Documents/AI-ML-Practice/IsaacLab/logs/rsl_rl/anymal_c_flat/2025-10-22_13-59-16/exported/policy.pt"

# Parse arguments
parser = argparse.ArgumentParser(description="Replay policy on single robot")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_POLICY_PATH, help="Path to exported policy.pt")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of steps")
parser.add_argument("--output_file", type=str, default="isaaclab_log.csv", help="Output CSV file")
parser.add_argument("--no-logging", action="store_true", help="Disable CSV logging (just run the policy)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import csv
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import importlib
from datetime import datetime

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.assets import retrieve_file_path


def import_class_from_string(class_path: str):
    """Import a class from a string path like 'module.submodule:ClassName'"""
    module_path, class_name = class_path.rsplit(':', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def main():
    # Task name - adjust this for your specific task
    task_name = "Isaac-Velocity-Flat-Anymal-C-Direct-v0"
    
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    
    # Get environment config
    env_cfg_entry_point_str = gym.spec(task_name).kwargs.get("env_cfg_entry_point")
    print(f"[INFO] Config entry point: {env_cfg_entry_point_str}")
    
    # Import the config class and instantiate it
    env_cfg_class = import_class_from_string(env_cfg_entry_point_str)
    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = 1  # Force single environment
    
    # Calculate required episode length based on desired steps
    # Direct RL envs use decimation, so actual control dt might be different from sim dt
    # For ANYmal-C flat: sim.dt=0.005 (200Hz), decimation=4 -> control dt=0.02 (50Hz)
    
    # Get the actual control timestep
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'dt'):
        sim_dt = env_cfg.sim.dt
    else:
        sim_dt = 0.005  # Default for Isaac Lab
    
    if hasattr(env_cfg, 'decimation'):
        decimation = env_cfg.decimation
    else:
        decimation = 4  # Default for ANYmal-C
    
    control_dt = sim_dt * decimation
    required_time = args_cli.num_steps * control_dt * 1.2  # 20% buffer
    
    # Set episode length to accommodate all steps (use 2x buffer for safety)
    original_length = getattr(env_cfg, 'episode_length_s', None)
    env_cfg.episode_length_s = args_cli.num_steps * control_dt
    print(f"[INFO] Episode length: {original_length}s -> {env_cfg.episode_length_s:.1f}s")
    
    # Disable all terminations to prevent early resets
    # For Direct RL envs, we need to clear the termination terms dict
    if hasattr(env_cfg, 'terminations'):
        # Try to access the internal dict structure
        if hasattr(env_cfg.terminations, '__dict__'):
            term_dict = env_cfg.terminations.__dict__
            disabled_terms = []
            for key, value in list(term_dict.items()):
                if not key.startswith('_') and value is not None:
                    disabled_terms.append(key)
                    setattr(env_cfg.terminations, key, None)
            print(f"[INFO] Disabled terminations: {disabled_terms}")
        else:
            print(f"[WARN] Could not disable terminations - unknown config structure")
    
    # Also try to disable events that might cause resets
    if hasattr(env_cfg, 'events'):
        if hasattr(env_cfg.events, 'reset_scene_to_default'):
            env_cfg.events.reset_scene_to_default = None
            print(f"[INFO] Disabled reset_scene_to_default event")
        if hasattr(env_cfg.events, 'reset_robot_joints'):
            env_cfg.events.reset_robot_joints = None
            print(f"[INFO] Disabled reset_robot_joints event")
    
    # Disable curriculum if present
    if hasattr(env_cfg, 'curriculum'):
        env_cfg.curriculum = None
        print(f"[INFO] Disabled curriculum")
    
    # Create environment with config
    env = gym.make(task_name, cfg=env_cfg)
    
    # Get robot and device (Direct env uses _robot, not scene["robot"])
    robot = env.unwrapped._robot
    contact_sensor = getattr(env.unwrapped, "_contact_sensor", None)
    device = env.unwrapped.device
    
    print(f"[INFO] Robot joints: {robot.num_joints}")
    print(f"[INFO] Device: {device}")
    
    # Get foot body indices for contact force logging
    if contact_sensor is not None:
        feet_body_indices, feet_names = contact_sensor.find_bodies(".*FOOT")
        print(f"[INFO] Found feet at indices {feet_body_indices}: {feet_names}")
        # Extract short names (LF, LH, RF, RH) from full names (LF_FOOT, etc.)
        foot_short_names = [name.replace("_FOOT", "") for name in feet_names]
    else:
        print(f"[WARN] No contact sensor found - cannot log contact forces")
        feet_body_indices = []
        foot_short_names = []
    
    # Load policy
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading policy from: {checkpoint_path}")
    policy = torch.jit.load(checkpoint_path, map_location=device)
    policy.eval()
    
    # Setup CSV logging with timestamp
    if not args_cli.no_logging:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Insert timestamp before file extension
        output_file = Path(args_cli.output_file)
        output_path = output_file.parent / f"{output_file.stem}_{timestamp}{output_file.suffix}"
        header = [
            "step", "time",
            "base_pos_x", "base_pos_y", "base_pos_z",
            "base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
        ]
        for i in range(robot.num_joints):
            header.append(f"joint_pos_{i}")
        for i in range(robot.num_joints):
            header.append(f"joint_vel_{i}")
        # Add contact forces for each foot (order from find_bodies: LF, LH, RF, RH)
        if contact_sensor is not None and len(feet_body_indices) > 0:
            for foot_name in foot_short_names:
                for axis in ["x", "y", "z"]:
                    header.append(f"contact_force_{foot_name}_{axis}")
        
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        print(f"[INFO] Logging to: {output_path}")
    else:
        print(f"[INFO] CSV logging disabled")
    
    # Reset and run
    obs, _ = env.reset()
    
    # IMPORTANT: Set fixed velocity command for deterministic comparison
    # Direct RL envs don't use command manager, so we set _commands directly
    # Commands are in BODY FRAME: [lin_vel_x, lin_vel_y, ang_vel_z]
    # Set to [0.5, 0.0, 0.0] to match MuJoCo script
    fixed_command = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env.unwrapped._commands[:] = fixed_command
    print(f"[INFO] Fixed velocity command: [0.5, 0.0, 0.0] m/s, rad/s in BODY FRAME (matches MuJoCo)")
    
    dt = env.unwrapped.step_dt
    
    # Check episode length
    max_episode_length = getattr(env.unwrapped, 'max_episode_length', 'Unknown')
    episode_length_s = getattr(env_cfg, 'episode_length_s', 'Unknown')
    
    print(f"[INFO] Running {args_cli.num_steps} steps...")
    print(f"[INFO] Step dt: {dt:.4f}s")
    print(f"[INFO] Max episode length: {max_episode_length} steps")
    print(f"[INFO] Episode length config: {episode_length_s}s")
    print(f"[INFO] Total simulation time: {args_cli.num_steps * dt:.2f}s\n")
    # print(f"[DEBUG] Env Config: {env_cfg}")
    
    reset_count = 0
    last_base_pos = robot.data.root_pose_w[0].cpu().numpy()[:3]
    
    for step in range(args_cli.num_steps):
        # Policy inference
        with torch.no_grad():
            # Extract policy observation from dict (Direct RL envs return dict observations)
            if isinstance(obs, dict):
                policy_obs = obs["policy"]
            else:
                policy_obs = obs
            actions = policy(policy_obs)
        
        # Step environment (no auto-reset since we disabled terminations)
        obs, _, terminated, truncated, _ = env.step(actions)
        
        # Check if environment terminated or truncated
        if terminated or truncated:
            print(f"[WARN] Step {step}: Environment terminated={terminated}, truncated={truncated}")
            reset_count += 1
            # Re-apply fixed command after reset
            env.unwrapped._commands[:] = fixed_command
            print(f"[WARN] Re-applied fixed command after reset")
        
        # Log data
        if not args_cli.no_logging:
            base_pose = robot.data.root_pose_w[0].cpu().numpy()
            joint_pos = robot.data.joint_pos[0].cpu().numpy()
            joint_vel = robot.data.joint_vel[0].cpu().numpy()
            
            # Get contact forces for feet
            contact_forces = []
            if contact_sensor is not None and len(feet_body_indices) > 0:
                # Get net contact forces: shape is (num_envs, num_bodies, 3)
                net_forces = contact_sensor.data.net_forces_w[0, feet_body_indices, :].cpu().numpy()
                contact_forces = net_forces.flatten().tolist()  # Flatten (4, 3) to 12 values
            
            # Detect if robot position suddenly jumped (indicating a reset)
            current_pos = base_pose[:3]
            pos_change = np.linalg.norm(current_pos - last_base_pos)
            if pos_change > 1.0 and step > 0:  # More than 1m jump
                print(f"[WARN] Step {step}: Robot position jumped {pos_change:.2f}m - likely a reset!")
                print(f"        Last pos: {last_base_pos}, Current: {current_pos}")
                reset_count += 1
            last_base_pos = current_pos.copy()
            
            row = [
                step, step * dt,
                *base_pose[:3].tolist(),  # position
                *base_pose[3:].tolist(),  # quaternion (w, x, y, z)
                *joint_pos.tolist(),
                *joint_vel.tolist(),
                *contact_forces,  # contact forces (if available)
            ]
            csv_writer.writerow(row)
        else:
            # Still get base pose for progress display
            base_pose = robot.data.root_pose_w[0].cpu().numpy()
        
        # Progress
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{args_cli.num_steps} | Height: {base_pose[2]:.3f}m")
    
    if not args_cli.no_logging:
        csv_file.close()
    
    env.close()
    
    print(f"\n{'='*70}")
    if not args_cli.no_logging:
        print(f"[INFO] Complete! Data saved to: {output_path}")
    else:
        print(f"[INFO] Complete! (no logging)")
    print(f"[INFO] Total resets detected: {reset_count}")
    if reset_count > 0:
        print(f"[WARN] The robot reset {reset_count} times during playback.")
        print(f"       This suggests terminations weren't fully disabled.")
    else:
        print(f"[SUCCESS] No resets detected - continuous playback achieved!")
    print(f"{'='*70}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n[ERROR]", traceback.format_exc())
    finally:
        simulation_app.close()

