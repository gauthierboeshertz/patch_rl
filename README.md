# patch_rl

## Installation:

install miniconda

create environment with the yml file : `conda env create -f environment.yml`
the env will be named urlb, so run `conda activate urlb` to activate the environment.

## Scripts:
To run the betaVAE training on euler run:

`sbatch  -n 4  --cpus-per-task=2  --gpus=1 --time=4:00:00    --mem-per-cpu=20G --wrap="python3 -u -m scripts.train_patch_model"`

To run the dynamics training, first put the path of the encoder in the train_dynamics.yaml Then on euler run:
`sbatch  -n 4  --cpus-per-task=2  --gpus=1 --time=4:00:00    --mem-per-cpu=20G --wrap="python3 -u -m scripts.train_dynamics" `.

To run the agent for the expert transitions run: 

`sbatch  -n 4  --cpus-per-task=2  --gpus=1 --time=24:00:00    --mem-per-cpu=20G --wrap="python3 -u -m scripts.train_agent"`.

To run the batch agent to test the CoDA algorithm: 

`sbatch  -n 4  --cpus-per-task=2  --gpus=1 --time=24:00:00    --mem-per-cpu=20G --wrap="python3 -u -m scripts.train_batch_agent"`.
The parameter dataset.use_gt_mask can be put to true to generate the ground truth mask, when it is false you need to provide the path to the trained dynamics model.



