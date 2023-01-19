#!/bin/bash

for zeed in {1..4}
do
   sbatch  -n 3  --cpus-per-task=2 --gpus=1  --time=4:00:00   --mem-per-cpu=20G --wrap="python3 -m scripts.train_batch_agent dataset_path=/data/expert_visual_5000transitions_4_all_sprite_mover_True_4instantmoveno_targets.npz 'seed=${zeed}'"
    echo "Sleeping for 200 seconds"
    sleep 100
done


for zeed in {1..4}
do
   sbatch  -n 3  --cpus-per-task=2 --gpus=1  --time=4:00:00   --mem-per-cpu=10G --wrap="python3 -m scripts.train_batch_agent dataset_path=/data/expert_visual_1000transitions_4_all_sprite_mover_True_4instantmoveno_targets.npz 'seed=${zeed}'"
    echo "Sleeping for 200 seconds"
    sleep 100
done

for zeed in {1..4}
do
   sbatch  -n 3  --cpus-per-task=2 --gpus=1  --time=4:00:00   --mem-per-cpu=10G --wrap="python3 -m scripts.train_batch_agent dataset_path=/data/expert_visual_500transitions_4_all_sprite_mover_True_4instantmoveno_targets.npz 'seed=${zeed}'"
    echo "Sleeping for 200 seconds"
    sleep 100
done

for zeed in {1..4}
do
   sbatch  -n 3  --cpus-per-task=2 --gpus=1  --time=4:00:00   --mem-per-cpu=10G --wrap="python3 -m scripts.train_batch_agent dataset_path=/data/expert_visual_100transitions_4_all_sprite_mover_True_4instantmoveno_targets.npz 'seed=${zeed}'"
    echo "Sleeping for 200 seconds"
    sleep 100
done
