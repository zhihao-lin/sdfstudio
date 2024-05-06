#!/bin/sh
#
#SBATCH --job-name=stump_norm_0
#SBATCH --output=/projects/bcrp/cl121/sdfstudio/zhi-hao/scripts/scannetpp/debug.out
#SBATCH --error=/projects/bcrp/cl121/sdfstudio/zhi-hao/scripts/scannetpp/debug.err
#
#SBATCH --account=bcrp-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=2-0:00
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
#SBATCH --mail-user=cl121@illinois.edu
#SBATCH --mail-type=ALL

source /u/cl121/.bashrc

conda activate sdfstudio
cd /scratch/bcrp/cl121/sdfstudio

# stump-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240505_stump_norm_0 \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/stump \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False