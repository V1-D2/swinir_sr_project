#!/bin/bash
#SBATCH --job-name=temp_swinir_sr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=72:00:00

echo "============================================"
echo "Temperature SwinIR Training Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 64GB per GPU"
echo "============================================"

# Set environment variables to suppress nvidia-smi warnings
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Install required packages if not already installed
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user basicsr opencv-python timm facexlib gfpgan tqdm matplotlib

# Verify environment in TensorFlow container
echo "Testing environment in TensorFlow container:"
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'❌ OpenCV: {e}')
try:
    import basicsr
    print(f'✅ BasicSR: {basicsr.__version__}')
except Exception as e:
    print(f'❌ BasicSR: {e}')
"

echo "============================================"
echo "Checking data directory..."
echo "Data should be available at: /scratch/tmp/data"
ls -la /scratch/tmp/data

echo "============================================"
echo "Starting Temperature SwinIR training with TensorFlow container:"

# Change to project directory
cd $HOME/swinir_sr_project

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run Temperature SR training in TensorFlow container
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/swinir_sr_project:$HOME/swinir_sr_project \
    --bind /scratch/tmp/data:/scratch/tmp/data \
    --env PYTHONPATH=$HOME/swinir_sr_project:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python train_temperature_sr.py \
    --data_dir /scratch/tmp/data \
    --output_dir ./experiments/temperature_swinir_rect_2nd_generation \
    --scale_factor 4 \
    --patch_height 512 \
    --patch_width 128 \
    --batch_size 8 \
    --num_epochs 70 \
    --lr 5e-4
    --max_files 100

echo "============================================"
echo "Temperature SwinIR Training Job Finished: $(date)"
echo "============================================"