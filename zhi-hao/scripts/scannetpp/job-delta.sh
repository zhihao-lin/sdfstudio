#!/bin/sh
#
#SBATCH --job-name=empty
#SBATCH --output=/projects/bcrp/cl121/sdfstudio/zhi-hao/scripts/scannetpp/debug.out
#SBATCH --error=/projects/bcrp/cl121/sdfstudio/zhi-hao/scripts/scannetpp/debug.err
#
#SBATCH --account=bcrp-delta-gpu
#SBATCH --partition=gpuA40x4
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
