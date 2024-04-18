#!/bin/bash

#SBATCH --job-name=sdfn-spp
#SBATCH --partition=shenlong2
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --mail-type=end
#SBATCH --mail-user=xiahongchi@sjtu.edu.cn
#SBATCH --output=logs/sdfn-spp.out
#SBATCH --error=logs/sdfn-spp.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=50G

nvidia-smi
pwd
module load gcc/11.2.0
module load git
source ~/.bashrc
source activate psdf

#python scripts/train.py nerfacto \
#    --vis wandb \
#    --pipeline.model.predict-normals True \
#    --pipeline.model.use_semantics True \
#    --experiment-name scannetpp-n \
#    panoptic-data \
#    --data /home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/plift/ \
#    --panoptic_data True \
#    --mono_data False \
#    --panoptic_segment True


#python scripts/train.py bakedsdf --vis wandb \
#    --output-dir outputs/scannetpp --experiment-name 240415_scannetpp_bakedsdf-colmap_sdfstudio \
#    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
#    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
#    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
#    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
#    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
#    panoptic-data \
#    --data /home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/plift/ \
#    --panoptic_data False \
#    --mono_data False \
#    --panoptic_segment False



python scripts/train.py bakedsdf --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240417_scannetpp_bakedsdf-colmap_sdfstudio_2e67a32314_normal_mono \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data /home/hongchix/scratch/scannetpp/2e67a32314/ \
    --panoptic_data False \
    --mono_normal_data True \
    --panoptic_segment False


python scripts/extract_mesh.py --load-config ./outputs/scannetpp/240415_scannetpp_bakedsdf-colmap_sdfstudio_normal_mono/bakedsdf/2024-04-15_094639/config.yml \
    --output-path ./outputs/scannetpp/240415_scannetpp_bakedsdf-colmap_sdfstudio_normal_mono/bakedsdf/2024-04-15_094639/mesh.ply \
    --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
    --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True