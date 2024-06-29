#!/usr/bin/env bash
#SBATCH --job-name=joint_training
#SBATCH --output=logs/joint_training_%j.log
#SBATCH --error=logs/joint_training_%j.err
#SBATCH --ntasks=1
#SBATCH --mail-user=sentanoe@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -e # Good Idea to stop operation on first error.

cd ~/joint_training          # navigate to the directory if necessary
source activate env_graph2route
srun python run_joint.py  --seed=2022 --spatial_encoder='gcn' --temporal_encoder='gru'       # python jobs require the srun command to work