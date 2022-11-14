#!/bin/bash

#SBATCH -n 5
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=10240

python3 gather_env_transitions.py --discrete_all_sprite_mover --random_init_places --num_sprites 4 --visual_obs --num_transitions 20000 --num_action_repeat 4

