#!/bin/bash
#SBATCH --job-name=test_swinir_sr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=2:00:00

echo "============================================"
echo "Temperature SwinIR Testing Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Set environment variables
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Change to project directory
cd $HOME/swinir_sr_project

# Run testing
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/swinir_sr_project:$HOME/swinir_sr_project \
    --env PYTHONPATH=$HOME/swinir_sr_project:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_temperature_sr.py \
    --model_path ./experiments/temperature_swinir_rect/checkpoints/best.pth \
    --input_npz ./test/single_amsr2_image.npz \
    --output_dir ./test_results \
    --num_samples 1 \
    --save_comparison \
    --scale_factor 4 \
    --patch_size 160 \
    --window_size 8

echo "============================================"
echo "Temperature SwinIR Testing Job Finished: $(date)"
echo "============================================"