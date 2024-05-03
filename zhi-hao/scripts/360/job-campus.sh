#!/bin/bash
#
#SBATCH --job-name=stump
#SBATCH --output=/projects/perception/personals/zhihao/sdfstudio/outputs/out.txt
#
#SBATCH --mail-user=cl121@illinois.edu
#SBATCH --mail-type=ALL
#
#SBATCH --partition=shenlong2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
#
#SBATCH --chdir=/home/cl121

source ~/.bashrc
export PATH=/projects/perception/personals/zhihao/miniconda3/envs/sdfstudio/bin:$PATH
export LD_LIBRARY_PATH=/projects/perception/personals/zhihao/miniconda3/envs/sdfstudio/lib:$LD_LIBRARY_PATH

conda activate sdfstudio
cd /projects/perception/personals/zhihao/sdfstudio

# stump
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_stump_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    mipnerf360-data --data data/colmap_sdfstudio/stump \
    --center-poses False --orientation-method none --auto-scale-poses False