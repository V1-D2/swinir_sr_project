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
echo "============================================"

# Set environment variables
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd $HOME/swinir_sr_project

# Install required packages if not already installed
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/swinir_sr_project:$HOME/swinir_sr_project \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user torch torchvision torchaudio timm opencv-python tqdm tensorboard

# Verify installation
echo "Verifying environment..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/swinir_sr_project:$HOME/swinir_sr_project \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
import cv2
print(f'OpenCV: {cv2.__version__}')
import timm
print(f'Timm: {timm.__version__}')
"

echo "============================================"
echo "Starting Temperature SwinIR SR training..."

# Run training
# Run training
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/swinir_sr_project:$HOME/swinir_sr_project \
    --bind $HOME/temperature_sr_project/data:$HOME/temperature_sr_project/data \
    --env PYTHONPATH=$HOME/swinir_sr_project:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python train_temperature_sr.py \
    --data_dir $HOME/temperature_sr_project/data \
    --output_dir ./experiments/temperature_swinir_x4 \
    --scale_factor 4 \
    --patch_size 128 \
    --batch_size 8 \
    --num_epochs 110 \
    --lr 2e-4

echo "============================================"
echo "Temperature SwinIR Training Job Finished: $(date)"
echo "============================================"