#!/bin/bash

for zeed in {1..4}
do
   sbatch  -n 5  --cpus-per-task=1 --gpus=1  --time=24:00:00   --mem-per-cpu=40G --wrap="python3 -m scripts.train_batch_agent 'seed=${zeed}'"
    echo "Sleeping for 200 seconds"
    sleep 100
done
