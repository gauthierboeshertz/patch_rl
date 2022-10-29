import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

from stable_baselines3.common.utils import set_random_seed
set_random_seed(0)


from progress.bar import Bar
import os
from src.patch_contrastive_model import PatchContrastiveModel
from src.datasets import SequenceImageTransitionDataset
from hydra.utils import get_original_cwd, to_absolute_path

logger = logging.getLogger(__name__)

def train_epoch(model, optimizer, dataloader):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        loss = model(batch)
        loss.backward()
        epoch_loss += loss.item()
        
        optimizer.step()
        model.update_targets()
        
    return epoch_loss/len(dataloader)

def val_epoch(model,dataloader):
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            loss = model(batch)
            epoch_loss += loss.item()
            
    return epoch_loss/len(dataloader)
                


def train(model, optimizer, train_dataloader, val_dataloader, num_epochs):
    info_bar = Bar('Training', max=num_epochs)
    min_val_loss = 100000
    
    for epoch in range(num_epochs):
        spr_train_loss = train_epoch(model,optimizer, train_dataloader)
        spr_val_loss = val_epoch(model,val_dataloader)         
        short_epoch_info = "Epoch: {},  train Loss: {}, Val Loss: {}".format(epoch,spr_train_loss,spr_val_loss )   
        
        epoch_info = f"Epoch: {epoch},TRAIN : SPR Loss: {spr_train_loss} ||   VAL : SPR Loss: {spr_val_loss}"
        logger.info(epoch_info)
        val_loss = spr_val_loss 
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        Bar.suffix = short_epoch_info
        info_bar.next()
    info_bar.finish()
    return model


@hydra.main(config_path="configs", config_name="patch_contrastive")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else "select_move"),config["env"]["random_init_places"])
    dataset = SequenceImageTransitionDataset(data_path=data_path)
    action_dim = dataset[0][1].shape[1]
    
    config["model"]["dynamics_dict"]["action_dim"] = action_dim
    model = PatchContrastiveModel(**config["model"])
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    train_dataloader = DataLoader(train_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=True,num_workers=config["train_loop"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=False,num_workers=config["train_loop"]["num_workers"])
    
    optimizer = torch.optim.Adam(model.parameters(),lr=config["train_loop"]["lr"])

    train(model,optimizer,train_dataloader,val_dataloader,config["train_loop"]["num_epochs"])
    print("Training Done")
    
    
if __name__ == "__main__":

    main()
    
