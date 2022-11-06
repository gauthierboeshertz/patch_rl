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
from src.dynamics_model import AttentionDynamicsModel
from src.datasets import SequenceImageTransitionDataset
from hydra.utils import get_original_cwd, to_absolute_path

logger = logging.getLogger(__name__)

def train_epoch(model, optimizer, dataloader):
    model.train()
    dyn_loss_epoch = 0
    byol_loss_epoch = 0
    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        loss, dyn_loss, byol_loss = model(batch)
        loss.backward()
        
        dyn_loss_epoch += dyn_loss.item()
        byol_loss_epoch += byol_loss.item()
        optimizer.step()
        model.update_targets()
        
    return dyn_loss_epoch/len(dataloader), byol_loss_epoch/len(dataloader)

def val_epoch(model,dataloader):
    model.eval()
    
    dyn_loss_epoch = 0
    byol_loss_epoch = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            loss, dyn_loss, byol_loss = model(batch)
            dyn_loss_epoch += dyn_loss.item()
            byol_loss_epoch += byol_loss.item()
            
    return dyn_loss_epoch/len(dataloader), byol_loss_epoch/len(dataloader)
                


def train(model, optimizer, train_dataloader, val_dataloader, num_epochs):
    info_bar = Bar('Training', max=num_epochs)
    min_val_loss = 100000
    
    for epoch in range(num_epochs):
        train_dyn_loss, train_byol_loss = train_epoch(model,optimizer, train_dataloader)
        val_dyn_loss, val_byol_loss = val_epoch(model,val_dataloader)         
        short_epoch_info = "Epoch: {},  train Loss: {}, Val Loss: {}".format(epoch,train_dyn_loss+train_byol_loss,val_dyn_loss+ val_byol_loss )   
        
        epoch_info = f"Epoch: {epoch},TRAIN : DYN Loss: {train_dyn_loss} BYOL LOSS: {train_byol_loss} ||   VAL : DYN Loss: {val_dyn_loss} BYOL LOSS: {val_byol_loss}"
        logger.info(epoch_info)
        val_loss = val_dyn_loss + (val_byol_loss*model.byol_loss_weight)
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            model.save_networks()

        Bar.suffix = short_epoch_info
        info_bar.next()
    info_bar.finish()
    return model


@hydra.main(config_path="configs", config_name="train_dynamics")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"])
    dataset = SequenceImageTransitionDataset(data_path=data_path)
    action_dim = dataset[0][1].shape[1]
    
    config["model"]["dynamics"]["action_dim"] = action_dim
    model = AttentionDynamicsModel(**config["model"])
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    train_dataloader = DataLoader(train_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=True,num_workers=config["train_loop"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=False,num_workers=config["train_loop"]["num_workers"])
    
    params_for_optim = model.get_trainable_params()
    optim_list = []
    for k in params_for_optim:
        optim_list.append({"params":params_for_optim[k],"lr":config["train_loop"]["lr"]})
    optimizer = torch.optim.Adam(optim_list)

    train(model,optimizer,train_dataloader,val_dataloader,config["train_loop"]["num_epochs"])
    print("Training Done")
    
    
if __name__ == "__main__":

    main()
    
