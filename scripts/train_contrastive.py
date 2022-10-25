import argparse
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from datasets import RecursiveTransitionDataset
from networks import *
from contrastive_model import ContrastiveModel

def main(config):
    
    print("Training with config: {}".format(config))
    
    assert config["replay_dir"] is not None, "Replay directory is not specified"
    
    dataset = RecursiveTransitionDataset(config["replay_dir"],num_replays=config["num_replays"],num_steps=config["num_steps"])
    action_dim = dataset[0][1].shape[0]
    
    model = ContrastiveModel(action_dim=action_dim,
                             encoder_model_name=config["encoder_model_name"],
                             device=config["device"],
                             lr=config["lr"],
                             seed=config["seed"])
                             
    
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    train_dataloader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True,num_workers=config["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=False,num_workers=config["num_workers"])

    model.train(train_dataloader,val_dataloader,config["num_epochs"])
    print("Training Done")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--num_replays', type=int, default=1000)
    parser.add_argument('--encoder_model_name', type=str, default="simple_cnn")
    parser.add_argument('--replay_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--num_steps',type=int, default=5)
    parser.add_argument('--num_workers',type=int, default=0)

    main(vars(parser.parse_args()))
    
