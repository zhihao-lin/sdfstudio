#!/bin/sh
#
#SBATCH --job-name=garden
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

# garden
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240501_garden_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    mipnerf360-data --data ../datasets/colmap_sdfstudio/garden_4 \
    --center-poses False --orientation-method none --auto-scale-poses False
