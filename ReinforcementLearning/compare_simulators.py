#!/usr/bin/env python3
"""
AnyMal C Simulator Comparison: MuJoCo vs IsaacSim

Compares joint trajectories and other metrics from the same policy executed
in both MuJoCo and IsaacLab simulators.

Usage:
    python compare_simulators.py --isaac isaaclab_log.csv --mujoco mujoco_single_robot_log.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Joint naming conventions
JOINT_NAMES_ISAAC = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"
]

JOINT_NAMES_MUJOCO = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE"
]

# Mapping from MuJoCo index to IsaacLab index
MUJOCO_TO_ISAAC_IDX = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


def load_data(isaac_file, mujoco_file):
    """Load and validate CSV data from both simulators."""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load IsaacSim data
    try:
        df_isaac = pd.read_csv(isaac_file)
        print(f"IsaacSim: {len(df_isaac)} rows, {len(df_isaac.columns)} columns")
        print(f"  Time range: {df_isaac['time'].min():.2f}s - {df_isaac['time'].max():.2f}s")
        
        # Check for contact force data
        has_contact = any('contact_force' in col for col in df_isaac.columns)
        print(f"  Contact forces: {'Available' if has_contact else 'Not available'}")
    except FileNotFoundError:
        print(f"IsaacSim log not found: {isaac_file}")
        return None, None
    
    # Load MuJoCo data
    try:
        df_mujoco = pd.read_csv(mujoco_file)
        print(f"MuJoCo: {len(df_mujoco)} rows, {len(df_mujoco.columns)} columns")
        print(f"  Time range: {df_mujoco['time'].min():.2f}s - {df_mujoco['time'].max():.2f}s")
        
        # Check for contact force data
        has_contact = any('contact_force' in col for col in df_mujoco.columns)
        print(f"  Contact forces: {'Available' if has_contact else 'Not available'}")
    except FileNotFoundError:
        print(f"MuJoCo log not found: {mujoco_file}")
        return None, None
    
    return df_isaac, df_mujoco


def align_data(df_isaac, df_mujoco, flip_mujoco_y=True, start_step=None, end_step=None):
    """Align datasets to common length and extract joint data."""
    print("\n" + "="*70)
    print("ALIGNING DATA")
    print("="*70)
    
    # Apply step filtering if specified
    if start_step is not None or end_step is not None:
        start_s = start_step if start_step is not None else 0
        end_s = end_step if end_step is not None else float('inf')
        
        # Filter both datasets by step number
        df_isaac = df_isaac[(df_isaac['step'] >= start_s) & (df_isaac['step'] <= end_s)].copy()
        df_mujoco = df_mujoco[(df_mujoco['step'] >= start_s) & (df_mujoco['step'] <= end_s)].copy()
        
        print(f"Step filter applied: {start_s} to {end_s}")
        print(f"  IsaacSim: {len(df_isaac)} rows remaining ({df_isaac['time'].min():.2f}s - {df_isaac['time'].max():.2f}s)")
        print(f"  MuJoCo: {len(df_mujoco)} rows remaining ({df_mujoco['time'].min():.2f}s - {df_mujoco['time'].max():.2f}s)")
    
    # Truncate to common length
    min_len = min(len(df_isaac), len(df_mujoco))
    max_time = min(df_isaac['time'].max(), df_mujoco['time'].max())
    print(f"Truncating to {min_len} steps (max time: {max_time:.2f}s)")
    
    df_isaac = df_isaac.iloc[:min_len].copy()
    df_mujoco = df_mujoco.iloc[:min_len].copy()
    
    # Coordinate system correction: flip Y-axis for MuJoCo to match IsaacLab
    if flip_mujoco_y:
        print(f"Applying Y-axis sign correction to MuJoCo data (coordinate system difference)")
        df_mujoco['base_pos_y'] = -df_mujoco['base_pos_y']
    
    # Extract joint data
    joint_pos_cols = [f"joint_pos_{i}" for i in range(12)]
    joint_vel_cols = [f"joint_vel_{i}" for i in range(12)]
    
    # IsaacLab data is in IsaacLab order
    isaac_joint_pos = df_isaac[joint_pos_cols].values
    isaac_joint_vel = df_isaac[joint_vel_cols].values
    
    # MuJoCo data needs reordering to IsaacLab order
    mujoco_joint_pos = df_mujoco[joint_pos_cols].values[:, MUJOCO_TO_ISAAC_IDX]
    mujoco_joint_vel = df_mujoco[joint_vel_cols].values[:, MUJOCO_TO_ISAAC_IDX]
    
    print(f"Joint position arrays: {isaac_joint_pos.shape}")
    print(f"Joint velocity arrays: {isaac_joint_vel.shape}")
    
    return df_isaac, df_mujoco, isaac_joint_pos, isaac_joint_vel, mujoco_joint_pos, mujoco_joint_vel


def compute_statistics(isaac_joint_pos, isaac_joint_vel, mujoco_joint_pos, mujoco_joint_vel):
    """Compute error statistics."""
    joint_pos_diff = isaac_joint_pos - mujoco_joint_pos
    joint_vel_diff = isaac_joint_vel - mujoco_joint_vel
    
    print("\n" + "="*70)
    print("JOINT POSITION DIFFERENCES (IsaacSim - MuJoCo)")
    print("="*70)
    print(f"{'Joint':<12} {'Mean (rad)':<12} {'Std (rad)':<12} {'Max Abs (rad)':<15} {'RMS (rad)':<12}")
    print("-"*70)
    
    for i, joint_name in enumerate(JOINT_NAMES_ISAAC):
        mean_diff = joint_pos_diff[:, i].mean()
        std_diff = joint_pos_diff[:, i].std()
        max_abs = np.abs(joint_pos_diff[:, i]).max()
        rms = np.sqrt((joint_pos_diff[:, i]**2).mean())
        print(f"{joint_name:<12} {mean_diff:>11.4f} {std_diff:>11.4f} {max_abs:>14.4f} {rms:>11.4f}")
    
    print("\n" + "="*70)
    print("JOINT VELOCITY DIFFERENCES (IsaacSim - MuJoCo)")
    print("="*70)
    print(f"{'Joint':<12} {'Mean (rad/s)':<14} {'Std (rad/s)':<14} {'Max Abs':<15} {'RMS (rad/s)':<12}")
    print("-"*70)
    
    for i, joint_name in enumerate(JOINT_NAMES_ISAAC):
        mean_diff = joint_vel_diff[:, i].mean()
        std_diff = joint_vel_diff[:, i].std()
        max_abs = np.abs(joint_vel_diff[:, i]).max()
        rms = np.sqrt((joint_vel_diff[:, i]**2).mean())
        print(f"{joint_name:<12} {mean_diff:>13.4f} {std_diff:>13.4f} {max_abs:>14.4f} {rms:>11.4f}")
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    pos_rms = np.sqrt((joint_pos_diff**2).mean())
    vel_rms = np.sqrt((joint_vel_diff**2).mean())
    print(f"Position - Mean RMS error: {pos_rms:.4f} rad ({np.rad2deg(pos_rms):.2f}°)")
    print(f"Position - Max abs error:  {np.abs(joint_pos_diff).max():.4f} rad ({np.rad2deg(np.abs(joint_pos_diff).max()):.2f}°)")
    print(f"Velocity - Mean RMS error: {vel_rms:.4f} rad/s")
    print(f"Velocity - Max abs error:  {np.abs(joint_vel_diff).max():.4f} rad/s")
    print("="*70)
    
    return joint_pos_diff, joint_vel_diff


def plot_joint_positions(time, isaac_joint_pos, mujoco_joint_pos, output_dir, timestamp):
    """Plot joint position comparisons."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, joint_name in enumerate(JOINT_NAMES_ISAAC):
        ax = axes[i]
        ax.plot(time, isaac_joint_pos[:, i], label='IsaacSim', linewidth=1.5, alpha=0.8)
        ax.plot(time, mujoco_joint_pos[:, i], label='MuJoCo', linewidth=1.5, alpha=0.8, linestyle='--')
        ax.set_title(f"{joint_name} Position", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Position (rad)", fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"joint_positions_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_joint_errors(time, joint_pos_diff, output_dir, timestamp):
    """Plot joint position errors."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, joint_name in enumerate(JOINT_NAMES_ISAAC):
        ax = axes[i]
        ax.plot(time, joint_pos_diff[:, i], linewidth=1.5, color='red', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_title(f"{joint_name} Error", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Error (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add RMS annotation
        rms = np.sqrt((joint_pos_diff[:, i]**2).mean())
        ax.text(0.98, 0.95, f'RMS: {rms:.4f}', 
                transform=ax.transAxes, 
                fontsize=8, 
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f"joint_position_errors_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_joint_velocities(time, isaac_joint_vel, mujoco_joint_vel, output_dir, timestamp):
    """Plot joint velocity comparisons."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, joint_name in enumerate(JOINT_NAMES_ISAAC):
        ax = axes[i]
        ax.plot(time, isaac_joint_vel[:, i], label='IsaacSim', linewidth=1.5, alpha=0.8)
        ax.plot(time, mujoco_joint_vel[:, i], label='MuJoCo', linewidth=1.5, alpha=0.8, linestyle='--')
        ax.set_title(f"{joint_name} Velocity", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Velocity (rad/s)", fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"joint_velocities_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_base_position(df_isaac, df_mujoco, output_dir, timestamp):
    """Plot base position comparisons."""
    time = df_isaac['time'].values
    isaac_base_pos = df_isaac[['base_pos_x', 'base_pos_y', 'base_pos_z']].values
    mujoco_base_pos = df_mujoco[['base_pos_x', 'base_pos_y', 'base_pos_z']].values
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['X', 'Y', 'Z']
    
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(time, isaac_base_pos[:, i], label='IsaacSim', linewidth=1.5, alpha=0.8)
        ax.plot(time, mujoco_base_pos[:, i], label='MuJoCo', linewidth=1.5, alpha=0.8, linestyle='--')
        ax.set_title(f"Base Position {label}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(f"Position {label} (m)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Compute RMS error
        diff = isaac_base_pos[:, i] - mujoco_base_pos[:, i]
        rms = np.sqrt((diff**2).mean())
        ax.text(0.02, 0.98, f'RMS Error: {rms:.4f} m', 
                transform=ax.transAxes, 
                fontsize=9, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f"base_position_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_contact_forces(df_isaac, df_mujoco, output_dir, timestamp):
    """Plot contact force comparisons."""
    # Detect available foot names from the columns
    # IsaacLab order is typically: LF, LH, RF, RH
    # MuJoCo order is typically: LF, RF, LH, RH
    foot_names = []
    for potential_foot in ["LF", "LH", "RF", "RH"]:
        if f"contact_force_{potential_foot}_z" in df_isaac.columns:
            foot_names.append(potential_foot)
    
    if len(foot_names) == 0:
        print(f"⚠ Skipping contact forces plot (data not available in CSV)")
        return
    
    # Check if all detected foot columns exist in both datasets
    contact_cols_exist = all(
        f"contact_force_{foot}_z" in df_isaac.columns and 
        f"contact_force_{foot}_z" in df_mujoco.columns
        for foot in foot_names
    )
    
    if not contact_cols_exist:
        print(f"⚠ Skipping contact forces plot (data not available in CSV)")
        return
    
    time = df_isaac['time'].values
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for foot_idx, foot_name in enumerate(foot_names):
        ax = axes[foot_idx]
        
        # Extract Z-axis (vertical) contact force
        isaac_force_z = df_isaac[f"contact_force_{foot_name}_z"].values
        mujoco_force_z = df_mujoco[f"contact_force_{foot_name}_z"].values
        
        ax.plot(time, isaac_force_z, label='IsaacSim', linewidth=1.5, alpha=0.8)
        ax.plot(time, mujoco_force_z, label='MuJoCo', linewidth=1.5, alpha=0.8, linestyle='--')
        ax.set_title(f"{foot_name} Foot Contact Force (Z)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Force (N)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        
        # Compute RMS error
        diff = isaac_force_z - mujoco_force_z
        rms = np.sqrt((diff**2).mean())
        ax.text(0.02, 0.98, f'RMS Error: {rms:.2f} N', 
                transform=ax.transAxes, 
                fontsize=9, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f"contact_forces_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_distribution(joint_pos_diff, output_dir, timestamp):
    """Plot error distribution analysis."""
    all_errors = joint_pos_diff.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Histogram
    axes[0].hist(all_errors, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_title("Joint Position Error Distribution", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Error (rad)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Statistics
    stats_text = f"Mean: {all_errors.mean():.4f} rad\n"
    stats_text += f"Std: {all_errors.std():.4f} rad\n"
    stats_text += f"Median: {np.median(all_errors):.4f} rad\n"
    stats_text += f"95th %%ile: {np.percentile(np.abs(all_errors), 95):.4f} rad"
    axes[0].text(0.98, 0.98, stats_text, 
                transform=axes[0].transAxes, 
                fontsize=9, 
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Box plot
    bp = axes[1].boxplot([joint_pos_diff[:, i] for i in range(12)], 
                          labels=JOINT_NAMES_ISAAC,
                          patch_artist=True,
                          showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_title("Joint Position Error by Joint", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Joint", fontsize=10)
    axes[1].set_ylabel("Error (rad)", fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"error_distribution_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_over_time(time, joint_pos_diff, output_dir, timestamp):
    """Plot error evolution over time."""
    cumulative_rms = np.sqrt(np.mean(joint_pos_diff**2, axis=1))
    cumulative_max = np.max(np.abs(joint_pos_diff), axis=1)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # RMS error
    axes[0].plot(time, cumulative_rms, linewidth=2, color='darkblue')
    axes[0].set_title("RMS Joint Position Error Over Time", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Time (s)", fontsize=10)
    axes[0].set_ylabel("RMS Error (rad)", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(time, 0, cumulative_rms, alpha=0.3)
    
    # Max error
    axes[1].plot(time, cumulative_max, linewidth=2, color='darkred')
    axes[1].set_title("Maximum Absolute Joint Position Error Over Time", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=10)
    axes[1].set_ylabel("Max Abs Error (rad)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(time, 0, cumulative_max, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"error_over_time_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Divergence analysis
    initial_rms = cumulative_rms[:100].mean() if len(cumulative_rms) > 100 else cumulative_rms.mean()
    final_rms = cumulative_rms[-100:].mean() if len(cumulative_rms) > 100 else cumulative_rms.mean()
    
    print(f"\nDivergence analysis:")
    print(f"  Initial RMS (first 100 steps): {initial_rms:.4f} rad")
    print(f"  Final RMS (last 100 steps):   {final_rms:.4f} rad")
    print(f"  Change: {((final_rms - initial_rms) / initial_rms * 100):+.1f}%")
    
    if final_rms > 1.5 * initial_rms:
        print("  ⚠ WARNING: Error is diverging!")
    elif final_rms < 0.75 * initial_rms:
        print("  Error is decreasing (convergence)")
    else:
        print("  Error remains stable")


def print_summary(df_isaac, df_mujoco, joint_pos_diff, joint_vel_diff, flip_applied=True):
    """Print comprehensive summary report."""
    isaac_base_pos = df_isaac[['base_pos_x', 'base_pos_y', 'base_pos_z']].values
    mujoco_base_pos = df_mujoco[['base_pos_x', 'base_pos_y', 'base_pos_z']].values
    base_pos_diff = isaac_base_pos - mujoco_base_pos
    
    # Check if contact force data is available
    has_contact_forces = any('contact_force' in col for col in df_isaac.columns) and \
                        any('contact_force' in col for col in df_mujoco.columns)
    
    print("\n" + "="*80)
    print("SIM-TO-SIM TRANSFER ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nDataset Info:")
    print(f"  Duration: {df_isaac['time'].max():.2f} seconds")
    print(f"  Steps:    {len(df_isaac)}")
    print(f"  Contact forces: {'Available' if has_contact_forces else 'Not available'}")
    print(f"  Y-axis flip: {'Applied (MuJoCo Y → -Y)' if flip_applied else 'Not applied'}")
    
    print(f"\nJoint Position Errors:")
    pos_rms = np.sqrt((joint_pos_diff**2).mean())
    print(f"  Mean RMS:     {pos_rms:.4f} rad ({np.rad2deg(pos_rms):.2f}°)")
    print(f"  Max absolute: {np.abs(joint_pos_diff).max():.4f} rad ({np.rad2deg(np.abs(joint_pos_diff).max()):.2f}°)")
    print(f"  Std dev:      {joint_pos_diff.std():.4f} rad ({np.rad2deg(joint_pos_diff.std()):.2f}°)")
    
    print(f"\nJoint Velocity Errors:")
    vel_rms = np.sqrt((joint_vel_diff**2).mean())
    print(f"  Mean RMS:     {vel_rms:.4f} rad/s")
    print(f"  Max absolute: {np.abs(joint_vel_diff).max():.4f} rad/s")
    
    print(f"\nBase Position Errors:")
    print(f"  X - RMS: {np.sqrt((base_pos_diff[:, 0]**2).mean()):.4f} m")
    print(f"  Y - RMS: {np.sqrt((base_pos_diff[:, 1]**2).mean()):.4f} m")
    print(f"  Z - RMS: {np.sqrt((base_pos_diff[:, 2]**2).mean()):.4f} m")
    
    print(f"\nWorst Joints (by RMS position error):")
    joint_rms = [np.sqrt((joint_pos_diff[:, i]**2).mean()) for i in range(12)]
    worst_joints = np.argsort(joint_rms)[::-1][:3]
    for rank, idx in enumerate(worst_joints, 1):
        print(f"  {rank}. {JOINT_NAMES_ISAAC[idx]}: {joint_rms[idx]:.4f} rad ({np.rad2deg(joint_rms[idx]):.2f}°)")
    
    print(f"\nBest Joints (by RMS position error):")
    best_joints = np.argsort(joint_rms)[:3]
    for rank, idx in enumerate(best_joints, 1):
        print(f"  {rank}. {JOINT_NAMES_ISAAC[idx]}: {joint_rms[idx]:.4f} rad ({np.rad2deg(joint_rms[idx]):.2f}°)")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    overall_rms_deg = np.rad2deg(pos_rms)
    if overall_rms_deg < 2.0:
        print("EXCELLENT: Sim-to-sim transfer is highly accurate (<2° RMS)")
    elif overall_rms_deg < 5.0:
        print("GOOD: Sim-to-sim transfer shows good agreement (<5° RMS)")
    elif overall_rms_deg < 10.0:
        print("⚠ MODERATE: Notable differences between simulators (5-10° RMS)")
    else:
        print("POOR: Significant sim-to-sim gap (>10° RMS)")
        print("\nTroubleshooting suggestions:")
        print("  - Verify joint ordering conversion")
        print("  - Check contact/friction parameters")
        print("  - Compare actuator models (position vs torque)")
        print("  - Verify timesteps and decimation match")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare AnyMal C simulators")
    parser.add_argument("--isaac", type=str, default="isaaclab_log.csv",
                        help="IsaacSim CSV log file")
    parser.add_argument("--mujoco", type=str, default="mujoco_single_robot_log.csv",
                        help="MuJoCo CSV log file")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for plots")
    parser.add_argument("--flip-mujoco-y", action="store_true", default=True,
                        help="Flip Y-axis sign for MuJoCo to match IsaacLab coordinate system (default: True)")
    parser.add_argument("--no-flip-mujoco-y", dest="flip_mujoco_y", action="store_false",
                        help="Don't flip Y-axis (use if coordinate systems already match)")
    parser.add_argument("--start-step", type=int, default=None,
                        help="Start step for analysis. Default: 0")
    parser.add_argument("--end-step", type=int, default=None,
                        help="End step for analysis. Default: use all data")
    parser.add_argument("--max-steps", type=int, default=None, dest="end_step",
                        help="Alias for --end-step (maximum steps to analyze)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_isaac, df_mujoco = load_data(args.isaac, args.mujoco)
    if df_isaac is None or df_mujoco is None:
        return
    
    # Align and extract
    df_isaac, df_mujoco, isaac_joint_pos, isaac_joint_vel, mujoco_joint_pos, mujoco_joint_vel = \
        align_data(df_isaac, df_mujoco, flip_mujoco_y=args.flip_mujoco_y, 
                   start_step=args.start_step, end_step=args.end_step)
    
    # Compute statistics
    joint_pos_diff, joint_vel_diff = compute_statistics(
        isaac_joint_pos, isaac_joint_vel, mujoco_joint_pos, mujoco_joint_vel
    )
    
    # Generate timestamp for all outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    print(f"Timestamp: {timestamp}\n")
    
    time = df_isaac['time'].values
    
    plot_joint_positions(time, isaac_joint_pos, mujoco_joint_pos, output_dir, timestamp)
    plot_joint_errors(time, joint_pos_diff, output_dir, timestamp)
    plot_joint_velocities(time, isaac_joint_vel, mujoco_joint_vel, output_dir, timestamp)
    plot_base_position(df_isaac, df_mujoco, output_dir, timestamp)
    plot_contact_forces(df_isaac, df_mujoco, output_dir, timestamp)
    plot_error_distribution(joint_pos_diff, output_dir, timestamp)
    plot_error_over_time(time, joint_pos_diff, output_dir, timestamp)
    
    # Print summary
    print_summary(df_isaac, df_mujoco, joint_pos_diff, joint_vel_diff, flip_applied=args.flip_mujoco_y)
    
    print(f"\nAll plots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

