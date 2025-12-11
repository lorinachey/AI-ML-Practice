#!/bin/bash
# Train a flat terrain policy for Anymal-C and export it for MuJoCo deployment

cd /home/lorin-cairo/Documents/AI-ML-Practice/IsaacLab

echo "======================================================================"
echo "Step 1/2: Training flat terrain policy..."
echo "======================================================================"

# Train with flat terrain task
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --num_envs 4096

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 2/2: Export policy for deployment...then play it back."
echo "======================================================================"

# Find the latest training run
LATEST_RUN=$(find logs/rsl_rl/anymal_c_flat -maxdepth 1 -type d -name "202*" | sort -r | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: Could not find training run directory"
    exit 1
fi

echo "Found training run: $LATEST_RUN"

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find "$LATEST_RUN" -name "model_*.pt" -type f | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No model checkpoints found"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

# Export policy by running play script (it auto-exports on load)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Anymal-C-Play-v0 \
    --num_envs 1 \
    --load_run $(basename "$LATEST_RUN") \
    --checkpoint "$LATEST_CHECKPOINT" \
    --disable_fabric

# Check if export succeeded
EXPORT_DIR="$LATEST_RUN/exported"
if [ -f "$EXPORT_DIR/policy.pt" ]; then
    echo ""
    echo "======================================================================"
    echo "SUCCESS! Policy exported!"
    echo "======================================================================"
    echo ""
    echo "Exported files:"
    echo "  - $EXPORT_DIR/policy.pt (TorchScript)"
    echo "  - $EXPORT_DIR/policy.onnx (ONNX)"
    echo ""
else
    echo ""
    echo "ERROR: Export failed. Check the output above for errors."
    exit 1
fi
