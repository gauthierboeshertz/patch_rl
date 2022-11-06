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
from src.vqvae_model import VQVAEModel
from src.datasets import SequenceImageTransitionDataset
from hydra.utils import get_original_cwd, to_absolute_path

logger = logging.getLogger(__name__)

def train_epoch(model, optimizer, dataloader):
    model.train()
    dyn_loss_epoch = 0
    vae_loss_epoch = 0

    total_loss = 0
    
    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()

        loss, vae_loss, dyn_loss = model(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        dyn_loss_epoch += float(dyn_loss)
        vae_loss_epoch += float(vae_loss)
        total_loss += float(loss)
        
        
    return total_loss/len(dataloader),vae_loss_epoch/len(dataloader), dyn_loss_epoch/len(dataloader)

def val_epoch(model,dataloader):
    model.eval()
    
    dyn_loss_epoch = 0
    vae_loss_epoch = 0
    total_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            loss,  vae_loss, dyn_loss = model(batch)
            dyn_loss_epoch += float(dyn_loss)
            vae_loss_epoch += float(vae_loss)
            total_loss += float(loss)
    return  total_loss/len(dataloader),vae_loss_epoch/len(dataloader),dyn_loss_epoch/len(dataloader)
                


def train(model, optimizer, train_dataloader, val_dataloader, num_epochs):
    info_bar = Bar('Training', max=num_epochs)
    min_val_loss = 100000
    
    for epoch in range(num_epochs):
        train_loss, train_vae_loss, train_dyn_loss = train_epoch(model,optimizer, train_dataloader)
        val_loss, val_vae_loss, val_dyn_loss = val_epoch(model,val_dataloader)         
        short_epoch_info = "Epoch: {},  train Loss: {}, Val Loss: {}".format(epoch,train_loss,val_loss )   
        
        epoch_info = f"Epoch: {epoch},TRAIN : DYN Loss: {train_dyn_loss} VAE LOSS: {train_vae_loss}  ||   VAL : DYN Loss: {val_dyn_loss} VAE LOSS: {val_vae_loss}"
        logger.info(epoch_info)
        model.save_networks("last")
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            model.save_networks("best_val")

        Bar.suffix = short_epoch_info
        info_bar.next()
    info_bar.finish()
    return model


@hydra.main(config_path="configs", config_name="train_vqvae")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""))
    dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=config["env"]["discrete_all_sprite_mover"])
    action_dim = dataset[0][1].shape[1]
    
    config["model"]["dynamics"]["action_dim"] = action_dim
    config["model"]["dynamics"]["discrete_actions"] =  config["env"]["discrete_all_sprite_mover"]

    model = VQVAEModel(**config["model"])
    
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
    
