#!/bin/bash

#SBATCH -n 10
#SBATCH --gpus=1
#SBATCH --gres=gpumem:14336m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=15240

python3 -m scripts.train_vqvae model.device=cuda

