"""
Sim-to-Sim Transfer: Isaac Lab Flat-Terrain Policy → MuJoCo 3.x

Deploys a policy trained in Isaac Lab (NVIDIA Isaac Sim) to MuJoCo simulator.
This demonstrates cross-simulator transfer with proper domain adaptation.

ROBOT: ANYmal-C quadruped robot
POLICY: Trained on flat terrain (48D observation space)
TESTED: mujoco_menagerie anymal_c scene.xml model

KEY IMPLEMENTATION DETAILS:
1. Joint Ordering Conversion
   - Isaac Lab: Groups by joint type [LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, ...]
   - MuJoCo: Groups by leg [LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, ...]
   - Script handles reordering automatically in both directions

2. Actuator Control
   - MuJoCo model uses built-in position actuators (kp=100)
   - Must initialize data.ctrl to match initial qpos (critical!)
   - Pass target positions directly to ctrl (not torques)

3. Timing Configuration
   - Simulation: 200 Hz (dt=0.005s) matches Isaac Lab
   - Control: 50 Hz (decimation=4) matches Isaac Lab policy frequency
   
4. Observation Space (48D for flat terrain)
   - Base: linear vel (3), angular vel (3), projected gravity (3)
   - Command: velocity command (3)
   - Joints: relative positions (12), velocities (12)
   - History: previous actions (12)
"""

import os
os.environ["MUJOCO_EGL"] = "1"  # Optional for headless rendering

import torch
import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.spatial.transform import Rotation
import time
import csv

# --------------------------------------------------------------------------- #
#                                CONFIGURATION                                #
# --------------------------------------------------------------------------- #
# These parameters match IsaacLab AnymalCFlatEnvCfg:
#   - sim.dt = 1/200 (0.005s = 200Hz)
#   - decimation = 4 (control at 50Hz)
#   - action_scale = 0.5
#   - static_friction = 1.0
#   - dynamic_friction = 1.0
#   - restitution = 0.0
# --------------------------------------------------------------------------- #

MODEL_PATH = "/home/lorin-cairo/Applications/mujoco-3.3.7/mujoco_menagerie/anybotics_anymal_c/scene.xml"

USE_HEIGHT_SCANNER = False   # False for flat policy (48D obs), True for rough (235D)
HEIGHT_SCANNER_RAYS = 187    # Number of height scan rays for rough terrain mode
DECIMATION = 4               # Control frequency decimation: 200Hz sim → 50Hz control (matches Isaac Lab)
ACTION_SCALE = 0.5           # Action scaling factor (from Isaac Lab config)
SIM_DT = 0.005               # Simulation timestep: 0.005s = 200 Hz (matches Isaac Lab)
SIM_TIME = 48.0              # Total simulation time in seconds

# Velocity command (INPUT to policy, not output)
# The policy is trained to TRACK this command - it doesn't generate it
# IMPORTANT: For fair sim-to-sim comparison, IsaacLab must use the SAME command!
# Default IsaacLab training uses commands sampled from ranges:
#   - lin_vel_x: [-1.0, 1.0] m/s
#   - lin_vel_y: [-1.0, 1.0] m/s  
#   - ang_vel_z: [-1.0, 1.0] rad/s
# For deterministic comparison, we use a fixed forward command:
VELOCITY_COMMAND = np.array([0.5, 0.0, 0.0])  # [v_x, v_y, omega_z] in m/s and rad/s

ALPHA = 0.2                  # Low-pass filter coefficient: 0.2 = 20% new + 80% old (smoother gait)
NUM_JOINTS = 12              # Total number of actuated joints (3 per leg × 4 legs)

# Logging configuration
ENABLE_LOGGING = True        # Enable data logging to CSV
LOG_FILE = "mujoco_single_robot_log.csv"  # Output CSV file for comparison with IsaacLab

if USE_HEIGHT_SCANNER:
    POLICY_PATH = "/home/lorin-cairo/Documents/AI-ML-Practice/IsaacLab/logs/rsl_rl/anymal_c_rough/2025-10-06_15-44-29/exported/policy.pt"
else:
    POLICY_PATH = "/home/lorin-cairo/Documents/AI-ML-Practice/IsaacLab/logs/rsl_rl/anymal_c_flat/2025-10-22_13-59-16/exported/policy.pt"

# --------------------------------------------------------------------------- #
#                                LOAD POLICY                                  #
# --------------------------------------------------------------------------- #

def load_isaaclab_policy(path):
    """Load TorchScript or checkpoint-based policy."""
    try:
        policy = torch.jit.load(path, map_location="cpu")
        print(" Loaded TorchScript policy")
        return policy
    except Exception as e:
        print(f"Warning: TorchScript load failed ({e}), trying torch.load...")
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("actor", "policy", "model_state_dict"):
                if key in checkpoint:
                    return checkpoint[key]
        return checkpoint

policy = load_isaaclab_policy(POLICY_PATH)
policy.eval()
print(" Policy loaded successfully!\n")

# Print configuration summary
print("=" * 70)
print("CONFIGURATION SUMMARY (Matching IsaacLab AnymalCFlatEnvCfg)")
print("=" * 70)
print(f"Simulation timestep:     {SIM_DT}s ({1/SIM_DT:.0f} Hz)")
print(f"Control decimation:      {DECIMATION}x ({1/(SIM_DT*DECIMATION):.0f} Hz control)")
print(f"Action scale:            {ACTION_SCALE}")
print(f"Simulation time:         {SIM_TIME}s")
print(f"Velocity command:        [{VELOCITY_COMMAND[0]}, {VELOCITY_COMMAND[1]}, {VELOCITY_COMMAND[2]}] m/s, rad/s")
print(f"Action filter (alpha):   {ALPHA}")
print(f"Height scanner:          {'Enabled' if USE_HEIGHT_SCANNER else 'Disabled'}")
print(f"Observation dimension:   {235 if USE_HEIGHT_SCANNER else 48}D")
print("=" * 70 + "\n")

# --------------------------------------------------------------------------- #
#                             LOAD MUJOCO MODEL                               #
# --------------------------------------------------------------------------- #

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Configure simulation parameters to match Isaac Lab
model.opt.timestep = SIM_DT
model.opt.gravity[:] = [0, 0, -9.81]
model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

# Find foot geom IDs for contact force extraction
# ANYmal-C feet are spherical geoms (size=0.03) at the end of each shank
# They're part of bodies: LF_SHANK, RF_SHANK, LH_SHANK, RH_SHANK
foot_body_names = ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]
foot_geom_ids = []

for body_name in foot_body_names:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        # Find spherical geoms belonging to this body (foot geoms are spheres with size ~0.03)
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                # Check if it's a sphere (type 2) and has priority=1 (foot geom)
                if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    if model.geom_priority[geom_id] == 1:  # Foot geoms have priority=1
                        foot_geom_ids.append(geom_id)
                        break
    except Exception as e:
        print(f"[WARN] Could not find foot geom for body {body_name}: {e}")

print(f"[INFO] Found {len(foot_geom_ids)} foot geom IDs: {foot_geom_ids}")
print(f"[INFO] Foot friction values (from XML):")
for idx, geom_id in enumerate(foot_geom_ids):
    print(f"       Foot {idx} (geom_id={geom_id}): {model.geom_friction[geom_id]}")

# TEST ONLY: Set foot friction to match IsaacLab config (Result: AnyMal faceplants)
# IsaacLab uses: static_friction=1.0, dynamic_friction=1.0, restitution=0.0
# for foot_geom_id in foot_geom_ids:
#     # Set friction (sliding, torsional, rolling) - ONLY for foot geoms
#     model.geom_friction[foot_geom_id, 0] = 1.0  # sliding friction (IsaacLab config, XML default is 0.8)
#     model.geom_friction[foot_geom_id, 1] = 0.005  # torsional friction
#     model.geom_friction[foot_geom_id, 2] = 0.0001  # rolling friction
#     # Set solref and solimp for contact dynamics
#     model.geom_solref[foot_geom_id, 0] = 0.02  # time constant for soft contact
#     model.geom_solref[foot_geom_id, 1] = 1.0   # damping ratio


print(f"[INFO] Updated friction for {len(foot_geom_ids)} foot geoms to match IsaacLab (1.0)")
print(f"       Other geoms retain their XML-specified friction values")

print("=" * 70)
print("MUJOCO MODEL INFO")
print("=" * 70)
print(f"qpos: {model.nq}, qvel: {model.nv}, actuators: {model.nu}")
print(f"Timestep: {model.opt.timestep:.4f}s  →  {1/model.opt.timestep:.0f} Hz sim rate")
print(f"Control every {DECIMATION} steps ({SIM_DT * DECIMATION:.3f}s, 50Hz)")
print("=" * 70)

# --------------------------------------------------------------------------- #
#                           INITIALIZE ROBOT POSE                             #
# --------------------------------------------------------------------------- #

# Joint ordering differs between Isaac Lab and MuJoCo:
# - MuJoCo: Groups by leg [LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE]
# - Isaac Lab: Groups by joint type [LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE]
# Leg naming: LF=Left Front, RF=Right Front, LH=Left Hind, RH=Right Hind

# Isaac Lab default standing pose
# HAA: 0.0, Front HFE: 0.4, Hind HFE: -0.4, Front KFE: -0.8, Hind KFE: 0.8
ISAAC_LAB_DEFAULT_JOINTS = np.array([
    0.0, 0.0, 0.0, 0.0,           # LF_HAA, LH_HAA, RF_HAA, RH_HAA
    0.4, -0.4, 0.4, -0.4,          # LF_HFE, LH_HFE, RF_HFE, RH_HFE  
    -0.8, 0.8, -0.8, 0.8           # LF_KFE, LH_KFE, RF_KFE, RH_KFE
])

def isaac_to_mujoco_order(isaac_joints):
    """
    Reorder joints from Isaac Lab format to MuJoCo format.
    
    Isaac Lab order (grouped by joint type):
        [LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE]
        Indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    MuJoCo order (grouped by leg):
        [LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE]
    """
    return np.array([
        isaac_joints[0], isaac_joints[4], isaac_joints[8],   # LF: HAA, HFE, KFE
        isaac_joints[2], isaac_joints[6], isaac_joints[10],  # RF: HAA, HFE, KFE
        isaac_joints[1], isaac_joints[5], isaac_joints[9],   # LH: HAA, HFE, KFE
        isaac_joints[3], isaac_joints[7], isaac_joints[11],  # RH: HAA, HFE, KFE
    ])

DEFAULT_JOINT_POS_MUJOCO = isaac_to_mujoco_order(ISAAC_LAB_DEFAULT_JOINTS)

# Standing pose configuration (MuJoCo order)
standing_qpos = np.array([
    0.0, 0.0, 0.6,          # Base position: x, y, z (0.6m height from Isaac Lab config)
    1.0, 0.0, 0.0, 0.0,     # Base orientation: quaternion (w, x, y, z) - upright
    *DEFAULT_JOINT_POS_MUJOCO  # Joint positions in MuJoCo order
])

# Choose starting pose: True = standing, False = lying down
START_STANDING = True

if START_STANDING:
    data.qpos[:] = standing_qpos
    data.qvel[:] = 0.0
    # CRITICAL: Initialize ctrl to match qpos! Without this, position actuators
    # will try to drive joints to zero, causing instability
    data.ctrl[:] = DEFAULT_JOINT_POS_MUJOCO
    mujoco.mj_forward(model, data)
    print(" Initialized to standing pose (Isaac Lab defaults)")
else:
    # Lying down pose (belly on ground, legs folded) - Isaac Lab order
    lying_joints_isaac = np.array([
        0.0, 0.0, 0.0, 0.0,        # HAA: All 0.0 (no abduction)
        1.0, -1.0, 1.0, -1.0,       # HFE: Front legs +1.0, Hind legs -1.0 (bent inward)
        -2.0, 2.0, -2.0, 2.0        # KFE: Front legs -2.0, Hind legs +2.0 (bent tight)
    ])
    lying_joints_mujoco = isaac_to_mujoco_order(lying_joints_isaac)
    
    data.qpos[0:3] = [0.0, 0.0, 0.35]      # Base position: x, y, z (lower height for lying)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Base orientation: quaternion (w, x, y, z) - upright
    data.qpos[7:19] = lying_joints_mujoco
    data.qvel[:] = 0.0
    # CRITICAL: Initialize ctrl to match qpos! (see comment above)
    data.ctrl[:] = lying_joints_mujoco
    mujoco.mj_forward(model, data)
    print(" Initialized to lying down pose")

previous_actions = np.zeros(NUM_JOINTS)
previous_joint_vel = np.zeros(NUM_JOINTS)
previous_torques = np.zeros(NUM_JOINTS)

# Debug: print initialization info
# print(f"Initial ctrl: {data.ctrl[:]}")
# print(f"Initial qpos (joints): {data.qpos[7:19]}")

# --------------------------------------------------------------------------- #
#                         CONTACT FORCE EXTRACTION                            #
# --------------------------------------------------------------------------- #

def extract_foot_contact_forces(model, data, foot_geom_ids):
    """
    Extract contact forces for each foot from MuJoCo contact data.
    Uses MuJoCo's mj_contactForce to get accurate contact forces.
    
    Returns:
        np.array: (4, 3) array of contact forces [fx, fy, fz] for each foot
                  Order: [LF, RF, LH, RH]
    """
    foot_forces = np.zeros((len(foot_geom_ids), 3))
    
    # Iterate through all active contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Check if either geom in contact is a foot
        for foot_idx, foot_geom_id in enumerate(foot_geom_ids):
            if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                # Use MuJoCo's contact force calculation
                # This returns forces in contact frame: [normal, tangent1, tangent2, tor1, tor2, tor3]
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i, c_array)
                
                # Get contact frame (columns are [normal, tangent1, tangent2])
                contact_frame = contact.frame.reshape(3, 3)
                
                # Transform contact forces from contact frame to world frame
                # Force in contact frame: [normal, tangent1, tangent2]
                force_contact_frame = c_array[0:3]
                force_world = contact_frame @ force_contact_frame
                
                # Accumulate force for this foot (multiple contact points may exist)
                foot_forces[foot_idx] += force_world
    
    return foot_forces

# --------------------------------------------------------------------------- #
#                              MAIN SIMULATION LOOP                           #
# --------------------------------------------------------------------------- #
# Control flow each step:
# 1. Read MuJoCo state (qpos, qvel) in MuJoCo joint order
# 2. Reorder joints from MuJoCo → Isaac Lab order
# 3. Construct 48D observation vector (Isaac Lab format)
# 4. Run policy inference → get 12D action (Isaac Lab order)
# 5. Convert action to target positions + reorder Isaac Lab → MuJoCo
# 6. Send target positions to MuJoCo actuators via data.ctrl
# 7. Step simulation DECIMATION times (4×5ms = 20ms control period)
# --------------------------------------------------------------------------- #

dt = model.opt.timestep
control_dt = dt * DECIMATION
steps = int(SIM_TIME / control_dt)

expected_obs_dim = 235 if USE_HEIGHT_SCANNER else 48
print(f"\nRunning {SIM_TIME}s sim → {steps} control steps")
print(f"Policy expects obs dim: {expected_obs_dim}")
print("=" * 70)

# Setup CSV logging
csv_file = None
csv_writer = None
if ENABLE_LOGGING:
    # Create CSV header
    header = [
        "step", "time",
        "base_pos_x", "base_pos_y", "base_pos_z",
        "base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
    ]
    # Add joint positions and velocities (in MuJoCo order to match IsaacLab)
    for i in range(NUM_JOINTS):
        header.append(f"joint_pos_{i}")
    for i in range(NUM_JOINTS):
        header.append(f"joint_vel_{i}")
    # Add contact forces for each foot [LF, RF, LH, RH]
    foot_names = ["LF", "RF", "LH", "RH"]
    for foot_name in foot_names:
        for axis in ["x", "y", "z"]:
            header.append(f"contact_force_{foot_name}_{axis}")
    
    csv_file = open(LOG_FILE, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    print(f"\nLogging enabled: {LOG_FILE}")
    print(f"CSV columns: {len(header)}\n")

with viewer.launch_passive(model, data) as v:
    for control_step in range(steps):
        # --- Observation Construction ---
        # Get base orientation and compute rotation matrix (world to body frame)
        base_quat = data.qpos[3:7]  # MuJoCo quaternion format: (w, x, y, z)
        # Convert to scipy format (x, y, z, w) and get rotation matrix transpose
        R_wb = Rotation.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix().T

        # Transform velocities and gravity to body frame
        base_lin_vel_body = R_wb @ data.qvel[0:3]
        base_ang_vel_body = R_wb @ data.qvel[3:6]
        projected_gravity_body = R_wb @ np.array([0, 0, -1])

        # Get joint states from MuJoCo (in MuJoCo order)
        joint_pos_mujoco = data.qpos[7:7+NUM_JOINTS]
        joint_vel_mujoco = data.qvel[6:6+NUM_JOINTS]
        
        # Convert MuJoCo order to Isaac Lab order for policy observation
        # MuJoCo indices:  [0, 1, 2,  3, 4, 5,  6, 7, 8,  9, 10, 11]
        # MuJoCo joints:   [LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE]
        # Isaac Lab order: [LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE]
        # Mapping: MuJoCo[0,6,3,9, 1,7,4,10, 2,8,5,11] → Isaac Lab[0-11]
        mujoco_to_isaac_idx = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11])
        joint_pos_isaac = joint_pos_mujoco[mujoco_to_isaac_idx]
        joint_vel_isaac = joint_vel_mujoco[mujoco_to_isaac_idx]
        
        # Debug: verify joint ordering conversion
        # if control_step % 100 == 0:
        #     print(f"Joint pos MuJoCo (first 3): {joint_pos_mujoco[:3]}")
        #     print(f"Joint pos Isaac (first 3): {joint_pos_isaac[:3]}")
        
        # Compute relative joint positions (deviation from default pose)
        joint_pos_rel = joint_pos_isaac - ISAAC_LAB_DEFAULT_JOINTS

        # # Optional height scanner for rough terrain (SIMULATED DATA)
        # if USE_HEIGHT_SCANNER:
        #     # Generate synthetic height measurements (ground-relative, not robot-relative)
        #     # Assume flat ground at z=0, measure vertical distance from current base height
        #     base_height = data.qpos[2]
        #     ground_level = 0.0
            
        #     # Base height difference (robot base to ground)
        #     nominal_height = base_height - ground_level
            
        #     # Add slight random terrain variations (±5cm) to simulate small bumps/dips
        #     # Each ray gets different noise to simulate spatial variation
        #     terrain_noise = np.random.uniform(-0.5, 0.5, HEIGHT_SCANNER_RAYS)
            
        #     # Height scanner returns distance from ground (positive = above ground)
        #     # Subtract nominal height so 0 means "at robot height level"
        #     height_data = terrain_noise  # Small variations around flat terrain
            
        #     # Clip to reasonable range (policy was trained with normalized heights)
        #     height_data = np.clip(height_data, -1.0, 1.0)
        if USE_HEIGHT_SCANNER:
            # --- Real Height Scanner via MuJoCo Raycasting ---
            base_pos = data.qpos[0:3].copy()
            base_quat = data.qpos[3:7]
            R_wb = Rotation.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()

            # Define ray origins and directions in local (body) frame
            # Rays form a circle under the robot + one in the center (like Isaac Lab)
            radius = 0.4  # scanner radius [m]
            angles = np.linspace(0, 2*np.pi, HEIGHT_SCANNER_RAYS, endpoint=False)
            origins_local = np.stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros_like(angles)
            ], axis=1)

            # Direction (in local frame): straight down
            dirs_local = np.tile(np.array([0, 0, -1.0]), (HEIGHT_SCANNER_RAYS, 1))

            height_data = np.zeros(HEIGHT_SCANNER_RAYS)
            for i in range(HEIGHT_SCANNER_RAYS):
                origin_world = base_pos + R_wb @ origins_local[i]
                direction_world = R_wb @ dirs_local[i]

                # Cast ray into scene
                geom_id_out = np.array([-1], dtype=np.int32)
                geomgroup = np.zeros(6, dtype=np.uint8)  # include all geom groups
                dist = mujoco.mj_ray(model, data,
                                    origin_world, direction_world,
                                    geomgroup,
                                    1,    # flg_static: 1 = collide with static geoms
                                    -1,   # bodyexclude: -1 = don't exclude any body
                                    geom_id_out)
                geom_id = geom_id_out[0]

                # If no hit, assign max range (say 2m below robot)
                if dist < 0 or dist > 2.0:
                    dist = 2.0

                # Convert to "height difference": positive = ground closer than nominal base height
                # Compute point of contact height (base_z - dist along direction z)
                ground_z = origin_world[2] + direction_world[2] * dist
                height_data[i] = ground_z - base_pos[2]  # relative to base height

            # Clip to normalized range (as in Isaac Lab)
            height_data = np.clip(height_data, -2.0, 2.0)
        else:
            height_data = np.array([])

        # Construct observation vector (Isaac Lab format)
        # Total: 48D for flat (3+3+3+3+12+12+12) or 235D for rough (+187 height scan)
        obs = np.concatenate([
            base_lin_vel_body,      # 3: linear velocity in body frame
            base_ang_vel_body,      # 3: angular velocity in body frame
            projected_gravity_body, # 3: gravity direction in body frame
            VELOCITY_COMMAND,       # 3: commanded velocity (x, y, yaw)
            joint_pos_rel,          # 12: joint positions relative to default (Isaac Lab order)
            joint_vel_isaac,        # 12: joint velocities (Isaac Lab order)
            height_data,            # 0 or 187: height scanner data (optional)
            previous_actions        # 12: previous actions (Isaac Lab order)
        ])
 
        # Debug: print full observation vector
        # print(f"Obs: {obs}")
 
        assert obs.shape[0] == expected_obs_dim, f"Obs mismatch: {obs.shape[0]} != {expected_obs_dim}"
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # --- Policy Inference ---
        with torch.no_grad():
            raw_action = policy(obs_tensor).squeeze(0).numpy()  # Returns 12D action (Isaac Lab order)
            # Debug: print raw policy output
            # print(f"Raw action: {raw_action}")

        # Apply low-pass filter for smoother gait (reduces jitter)
        action_isaac = ALPHA * raw_action + (1 - ALPHA) * previous_actions
        previous_actions = action_isaac

        # Convert action to absolute target joint positions
        # Action is relative to default pose, scaled by ACTION_SCALE
        target_joint_pos_isaac = ACTION_SCALE * action_isaac + ISAAC_LAB_DEFAULT_JOINTS
        
        # Reorder from Isaac Lab format to MuJoCo format
        target_joint_pos_mujoco = isaac_to_mujoco_order(target_joint_pos_isaac)
        
        # Debug: print target positions
        # if control_step % 100 == 0:
        #     print(f"Target positions (Isaac order, first 3): {target_joint_pos_isaac[:3]}")
        #     print(f"Target positions (MuJoCo order, first 3): {target_joint_pos_mujoco[:3]}")
        
        # MuJoCo model has built-in position actuators (kp=100)
        # Pass target joint positions directly to ctrl (not torques!)
        data.ctrl[:] = target_joint_pos_mujoco

        # --- Step simulation DECIMATION times ---
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
            v.sync()  # Sync viewer with data
            time.sleep(dt)

        # --- Log data to CSV ---
        if ENABLE_LOGGING and csv_writer is not None:
            # Base position
            base_pos = data.qpos[0:3]  # [x, y, z]
            # Base orientation (quaternion in MuJoCo format: w, x, y, z)
            base_quat = data.qpos[3:7]  # [w, x, y, z]
            # Joint positions and velocities (MuJoCo order)
            joint_pos = data.qpos[7:7+NUM_JOINTS]
            joint_vel = data.qvel[6:6+NUM_JOINTS]
            
            # Extract contact forces for all feet
            foot_contact_forces = extract_foot_contact_forces(model, data, foot_geom_ids)
            
            # Build data row
            row = [
                control_step,
                control_step * control_dt,
                *base_pos.tolist(),
                *base_quat.tolist(),
                *joint_pos.tolist(),
                *joint_vel.tolist(),
                *foot_contact_forces.flatten().tolist(),  # Flatten (4, 3) to 12 values
            ]
            csv_writer.writerow(row)

        # --- Debug every 50 control steps ---
        if control_step % 50 == 0:
            base_vel_xy = np.linalg.norm(base_lin_vel_body[:2])
            print(f"Step {control_step:04d} | Height: {data.qpos[2]:.3f} m | Vel: {base_vel_xy:.3f} m/s")
            
            # Debug: detailed state info
            # print(f"Data: {data} Model: {model}")
            # print(f"Height scan (first 5): {height_data[:5]}")
            # print(f"  Ctrl (first 3): {data.ctrl[:3]}")
            # print(f"  Qpos joints (first 3): {data.qpos[7:10]}")
            # print(f"  Joint pos error (first 3): {target_joint_pos_mujoco[:3] - data.qpos[7:10]}")
    
    print("\n" + "=" * 70)
    print(" Simulation finished.")
    print("Press ENTER to close the viewer...")
    print("=" * 70)
    input()

# Close CSV file
if ENABLE_LOGGING and csv_file is not None:
    csv_file.close()
    print(f"\n Data logged to: {LOG_FILE}")
    print(f" Total rows: {steps}")

print(" Viewer closed.")
