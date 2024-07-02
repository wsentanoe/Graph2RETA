#!/usr/bin/env bash
#SBATCH --job-name=Graph2RETA
#SBATCH --output=logs/Graph2RETA_%j.log
#SBATCH --error=logs/Graph2RETA_%j.err
#SBATCH --ntasks=1
#SBATCH --mail-user=sentanoe@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -e # Good Idea to stop operation on first error.

cd ~/Graph2RETA          # navigate to the directory if necessary
source activate env_graph2route
srun python run_joint.py  --seed=2022 --spatial_encoder='gcn' --temporal_encoder='tft' --num_epoch=10       # python jobs require the srun command to work