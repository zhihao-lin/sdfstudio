#!/bin/bash
#
#SBATCH --job-name=bakedsdf-mlp
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

# bakedsdf-mlp
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240422_office3_bakedsdf-mlp-non-normalized \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../../../datasets/scannetpp/data/4a1a3a7dc5/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False