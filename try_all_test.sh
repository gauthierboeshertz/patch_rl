data_path="/data/expert_visual_1000transitions_4_all_sprite_mover_True_4instantmoveno_targets.npz"
for zeed in {1..4}
do
   sbatch  -n 5  --cpus-per-task=1 --gpus=1  --time=24:00:00   --mem-per-cpu=40G --wrap="python3 -m scripts.train_batch_agent 'seed=${zeed} dataset_path=${data_path} '"
    echo "Sleeping for 200 seconds"
    sleep 50
done

data_path="/data/expert_visual_500transitions_4_all_sprite_mover_True_4instantmoveno_targets.npz"
for zeed in {1..4}
do
   sbatch  -n 5  --cpus-per-task=1 --gpus=1  --time=24:00:00   --mem-per-cpu=40G --wrap="python3 -m scripts.train_batch_agent 'seed=${zeed} dataset_path=${data_path} '"
    echo "Sleeping for 200 seconds"
    sleep 50
done

data_path="/data/expert_visual_100transitions_4_all_sprite_mover_True_4instantmoveno_targets.npz"
for zeed in {1..4}
do
   sbatch  -n 5  --cpus-per-task=1 --gpus=1  --time=24:00:00   --mem-per-cpu=40G --wrap="python3 -m scripts.train_batch_agent 'seed=${zeed} dataset_path=${data_path} '"
    echo "Sleeping for 200 seconds"
    sleep 50
done
