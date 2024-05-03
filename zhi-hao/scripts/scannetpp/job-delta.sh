#!/bin/sh
#
#SBATCH --job-name=empty
#SBATCH --output=/projects/bcrp/cl121/sdfstudio/zhi-hao/scripts/scannetpp/debug.out
#SBATCH --error=/projects/bcrp/cl121/sdfstudio/zhi-hao/scripts/scannetpp/debug.err
#
#SBATCH --account=bcrp-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=2-0:00
#SBATCH --mem=64GB
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

# empty
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_empty_bakedsdf-mlp \
    --trainer.load-dir outputs/scannetpp/240429_empty_bakedsdf-mlp/bakedsdf-mlp/2024-04-30_104016/sdfstudio_models \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/1c4b893630/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False
