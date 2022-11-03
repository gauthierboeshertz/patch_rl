#!/bin/bash

#SBATCH -n 5
#SBATCH --gpus=1
#SBATCH --gres=gpumem:14336m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10240

python3 -m scripts.train_patch_vae model.device=cuda

