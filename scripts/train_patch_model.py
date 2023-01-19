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
from src.patch_model import PatchModel
from src.datasets import SequenceImageTransitionDataset
from hydra.utils import get_original_cwd, to_absolute_path
from collections import defaultdict
import einops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

def train_epoch(model, optimizer, dataloader):
    model.train()

    total_loss = 0
    epoch_loss_dict = defaultdict(float)
    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()

        loss, loss_dict = model(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        for k in loss_dict:
            epoch_loss_dict[k] += float(loss_dict[k])
        total_loss += float(loss)
        
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= len(dataloader)
    return total_loss/len(dataloader), epoch_loss_dict

def val_epoch(model,dataloader):
    model.eval()
    
    epoch_loss_dict = defaultdict(float)
    total_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            loss,  loss_dict = model(batch)
            for k in loss_dict:
                epoch_loss_dict[k] += float(loss_dict[k])
            total_loss += float(loss)
            
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= len(dataloader)

    return  total_loss/len(dataloader), epoch_loss_dict
                


def train(model, optimizer, train_dataloader, val_dataloader, num_epochs,scheduler):
    info_bar = Bar('Training', max=num_epochs)
    min_val_loss = 100000
    
    for epoch in range(num_epochs):
        train_loss, train_loss_dict = train_epoch(model,optimizer, train_dataloader)
        val_loss, val_loss_dict = val_epoch(model,val_dataloader)        
        scheduler.step()
        short_epoch_info = "Epoch: {},  train Loss: {}, Val Loss: {}".format(epoch,train_loss,val_loss)   
        
        epoch_info = f"Epoch: {epoch}, TRAIN: "
        for k in train_loss_dict:
            epoch_info += f"{k}: {train_loss_dict[k]:.5f}, "
        epoch_info += "VAL: "
        for k in val_loss_dict:
            epoch_info += f"{k}: {val_loss_dict[k]:.5f}, "
        logger.info(epoch_info)
        model.save_networks("last")
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            model.save_networks("best_val")

        Bar.suffix = short_epoch_info
        info_bar.next()
    info_bar.finish()
    return model

def setup_model(config,num_actions):
    
    config["encoder_decoder"]["in_channels"] = (3 if (config["env"]["instant_move"] or config["env"]["discrete_all_sprite_mover"]) else 9)
    print(config["encoder_decoder"]["in_channels"])

    encoder_decoder = hydra.utils.instantiate(config["encoder_decoder"]).to(device)#PatchVAE(**patch_vae).to(self.device)
    
    config["inverse"]["encoder"]["in_channels"] = (config["encoder_decoder"]["embed_dim"] * 2) * (config["encoder_decoder"]["categorical_dim"] if "cat" in config["encoder_decoder"]["name"].lower()  else 1)
    config["inverse"]["mlp"]["input_size"] = config["inverse"]["encoder"]["channels"][-1]*encoder_decoder.num_patches
    config["inverse"]["mlp"]["output_size"] =  (16 if (config["env"]["discrete_all_sprite_mover"]) else 8)#config["dynamics"]["num_actions"] * config["dynamics"]["action_dim"]
    config["inverse"]["num_actions"] = num_actions
    config["inverse"]["discrete_action_space"] = config["env"]["discrete_all_sprite_mover"]
    
    print(config["inverse"])
    inverse_model = hydra.utils.instantiate(config["inverse"]).to(device)
    
    model = PatchModel(encoder_decoder=encoder_decoder,inverse=inverse_model,
                       vae_loss_weight=config["loss_weights"]["vae_loss_weight"],
                       inverse_loss_weight=config["loss_weights"]["inverse_loss_weight"],
                       device=device)
    return model

@hydra.main(config_path="configs", config_name="train_patch_model")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""), ("no_targets" if config["env"]["dont_show_targets"] else ""))
    dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=False,sequence_length=2)#config["env"]["discrete_all_sprite_mover"])
    num_actions = dataset.actions.shape[1]
    model = setup_model(config,num_actions)
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    train_dataloader = DataLoader(train_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=True,num_workers=config["train_loop"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=False,num_workers=config["train_loop"]["num_workers"])
    
    print("Starting Training")
    params_for_optim = model.get_trainable_params()
    optim_list = []
    for k in params_for_optim:
        optim_list.append({"params":params_for_optim[k],"lr":config["train_loop"]["lr"]})
    optimizer = torch.optim.Adam(optim_list)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["train_loop"]["scheduler_milestones"], gamma=0.2)
    train(model,optimizer,train_dataloader,val_dataloader,config["train_loop"]["num_epochs"],scheduler)
    print("Training Done")
    
    
if __name__ == "__main__":

    main()
    
