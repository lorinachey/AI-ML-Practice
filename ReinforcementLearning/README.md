# Sim-to-Sim Transfer: Isaac Lab to MuJoCo

> **Note:** This repository contains experimental research code. The implementations are under active development and may contain bugs or incomplete features. Use at your own discretion for research and educational purposes.

## Overview

This directory contains scripts for training and deploying reinforcement learning policies across different physics simulators, specifically transferring policies trained in NVIDIA Isaac Lab to MuJoCo 3.x. This work demonstrates cross-simulator policy transfer for legged locomotion on the ANYmal-C quadruped robot.

## Motivation

Sim-to-sim transfer serves as an important intermediate step toward sim-to-real deployment. By validating that policies trained in one simulator can successfully transfer to another, we can:

1. Verify that learned behaviors are not overfitting to simulator-specific artifacts
2. Identify and resolve domain adaptation challenges in a controlled environment
3. Leverage the complementary strengths of different simulators (e.g., Isaac Lab for parallel training, MuJoCo for fast inference)
4. Build confidence in the generalization capabilities of trained policies before real-world deployment

## Files

### `train_flat_policy.sh`

Training and export pipeline for flat terrain locomotion policies.

**Purpose:** Automates the complete workflow from training to policy export, producing a TorchScript model suitable for deployment in MuJoCo.

**Workflow:**
1. Trains an ANYmal-C policy using RSL-RL in Isaac Lab on flat terrain
2. Automatically locates the most recent training run and checkpoint
3. Exports the trained policy to TorchScript format via the play script
4. Validates successful export and provides deployment instructions

**Configuration:**
- Task: `Isaac-Velocity-Flat-Anymal-C-v0`
- Observation space: 48D (no height scanner)
- Training environments: 4096 parallel environments
- Export format: TorchScript (.pt) and ONNX (.onnx)

**Usage:**
```bash
cd /path/to/ReinforcementLearning
./train_flat_policy.sh
```

### `test_anymal_policy.py`

MuJoCo deployment script for Isaac Lab-trained policies.

**Purpose:** Deploys and evaluates TorchScript policies from Isaac Lab in the MuJoCo physics simulator, handling all necessary domain adaptation and interface translation.

**Key Implementation Details:**

#### 1. Joint Ordering Conversion
Isaac Lab and MuJoCo use fundamentally different joint orderings:
- **Isaac Lab:** Groups joints by type (all HAA joints, then all HFE joints, then all KFE joints)
  ```
  [LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE]
  ```
- **MuJoCo:** Groups joints by leg (all left-front joints, then right-front, etc.)
  ```
  [LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE]
  ```

The script implements bidirectional conversion functions to handle this mismatch transparently.

#### 2. Actuator Model Adaptation
Isaac Lab policies output target joint positions assuming PD control, while MuJoCo's anymal_c model includes built-in position actuators (kp=100). Critical implementation details:
- Target positions must be passed directly to `data.ctrl` (not torques)
- Initial `data.ctrl` must be initialized to match `data.qpos` to prevent instability
- Action scaling (0.5) matches the Isaac Lab training configuration

#### 3. Observation Space Reconstruction
The script reconstructs the 48-dimensional observation vector expected by the policy:
- Base state: linear velocity (3), angular velocity (3), projected gravity (3)
- Command: velocity command (3)
- Joint state: relative positions (12), velocities (12)
- History: previous actions (12)

All velocities and gravity vectors are transformed to the body frame using the appropriate rotation matrices.

#### 4. Temporal Alignment
Maintains identical timing to Isaac Lab:
- Simulation frequency: 200 Hz (dt = 0.005s)
- Control frequency: 50 Hz (decimation factor = 4)
- Low-pass filtering (alpha = 0.2) smooths policy outputs

**Usage:**
```bash
conda activate mujoco
cd /path/to/ReinforcementLearning
python test_anymal_policy.py
```

**Configuration:**
Edit the following parameters in the script header:
- `POLICY_PATH`: Path to exported TorchScript policy
- `VELOCITY_COMMAND`: Desired velocity command [v_x, v_y, omega_z]
- `START_STANDING`: Initial pose (True = standing, False = lying down)
- `SIM_TIME`: Simulation duration in seconds

## Technical Challenges and Solutions

### Challenge 1: Joint Ordering Mismatch
**Problem:** Incompatible joint orderings between simulators caused incorrect action execution.

**Solution:** Implemented index mapping arrays to convert between Isaac Lab (type-grouped) and MuJoCo (leg-grouped) orderings in both directions. This conversion occurs:
- When reading MuJoCo state for observation construction
- When sending actions from policy to MuJoCo actuators

### Challenge 2: Actuator Interface Differences
**Problem:** Initial implementation attempted torque control, but MuJoCo model uses position actuators.

**Solution:** Modified control interface to pass target positions directly to `data.ctrl`, matching MuJoCo's built-in PD controllers. Added critical initialization step to set initial control targets equal to initial joint positions.

### Challenge 3: Coordinate Frame Transformations
**Problem:** Quaternion representations differ between simulators.

**Solution:** Implemented explicit quaternion format conversion (MuJoCo's [w,x,y,z] to scipy's [x,y,z,w]) and proper rotation matrix construction for body-frame transformations.

## Results

The implementation successfully achieves sim-to-sim transfer with qualitatively similar locomotion behavior between Isaac Lab and MuJoCo. The robot maintains stable walking gaits and tracks velocity commands when deployed in MuJoCo using policies trained exclusively in Isaac Lab.

## Requirements

**Isaac Lab Environment:**
- NVIDIA Isaac Sim 4.5+
- Isaac Lab framework
- RSL-RL reinforcement learning library
- Python 3.11 with PyTorch

**MuJoCo Environment:**
- MuJoCo 3.x
- mujoco_menagerie (anybotics_anymal_c model)
- Python 3.x with PyTorch
- scipy (for rotation transformations)

## Future Work

Potential extensions include:
1. Rough terrain policy transfer (235D observation space with height scanner)
2. Domain randomization analysis to improve transfer robustness
3. Quantitative performance comparison between simulators
4. Extension to other quadruped platforms (Unitree Go1, Spot, etc.)
5. Real-world deployment following successful sim-to-sim validation

## References

- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- MuJoCo: https://mujoco.org/
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- ANYmal-C: https://www.anybotics.com/

## License

This code follows the licensing of the Isaac Lab framework and MuJoCo Menagerie assets.

