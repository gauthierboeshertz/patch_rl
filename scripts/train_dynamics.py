import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
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
from src.mask_utils import make_gt_causal_mask
from IPython import embed

logger = logging.getLogger(__name__)

def train_epoch(model, encoder_decoder,optimizer, dataloader,device):
    model.train()

    total_loss = 0
    epoch_loss_dict = defaultdict(float)
    for _, batch in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        loss, loss_dict = model.compute_loss(batch,encoder_decoder)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        for k in loss_dict:
            epoch_loss_dict[k] += float(loss_dict[k])
        total_loss += float(loss)
        
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= len(dataloader)
    return total_loss/len(dataloader), epoch_loss_dict

def val_epoch(model,encoder_decoder,dataloader,device):
    model.eval()
    
    epoch_loss_dict = defaultdict(float)
    total_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):

            loss,  loss_dict = model.compute_loss(batch,encoder_decoder)
            for k in loss_dict:
                epoch_loss_dict[k] += float(loss_dict[k])
            total_loss += float(loss)
            
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= len(dataloader)

    return  total_loss/len(dataloader), epoch_loss_dict
                


def train(model, encoder_decoder,optimizer, train_dataloader, val_dataloader, num_epochs,device,scheduler):
    info_bar = Bar('Training', max=num_epochs)
    min_val_loss = 100000
    
    for epoch in range(num_epochs):
        train_loss, train_loss_dict = train_epoch(model,encoder_decoder,optimizer, train_dataloader,device)
        val_loss, val_loss_dict = val_epoch(model,encoder_decoder,val_dataloader,device)   
        
        scheduler.step()
        
        short_epoch_info = "Epoch: {},  train Loss: {}, Val Loss: {}".format(epoch,train_loss,val_loss )   
        epoch_info = f"Epoch: {epoch}, TRAIN: "
        for k in train_loss_dict:
            epoch_info += f"{k}: {train_loss_dict[k]:.5f}, "
        epoch_info += "VAL: "
        for k in val_loss_dict:
            epoch_info += f"{k}: {val_loss_dict[k]:.5f}, "
        #epoch_info = f"Epoch: {epoch},TRAIN : DYN Loss: {train_dyn_loss} VAE LOSS: {train_vae_loss}  INV LOSS: {train_inv_loss}||   VAL : DYN Loss: {val_dyn_loss} VAE LOSS: {val_vae_loss} INV LOSS: {val_inv_loss}"
        logger.info(epoch_info)
        torch.save(model.state_dict(), f"last_dynamics.pt")
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"best_val_dynamics.pt")

        Bar.suffix = short_epoch_info
        info_bar.next()
    info_bar.finish()
    return model

def encodings_dataset(encoder_decoder, dataloader, save_obs=False,device="cpu"):
    
    all_obs = []
    obs_encodings = []
    actions = []
    rewards = []
    #if use_gt_masks:
    #    gt_masks = []
        
    for i, batch in enumerate(dataloader):
        obs,action,_,reward = batch
        obs = obs.to(device)
        action = action.to(device)
        t_obs = einops.rearrange(obs, "b t c h w -> (b t) c h w")
        encodings = encoder_decoder.get_encoding_for_dynamics(t_obs)
        encodings = einops.rearrange(encodings, "(b t n) c -> b t n c", b = action.shape[0],t=action.shape[1])
        
        """
        if use_gt_masks:
            t_gt_mask = []
            for t in range(action.shape[1]-1):
                t_gt_mask.append(make_gt_causal_mask(obs[:,t:t+2],action[:,t],patch_size=encoder_decoder.patch_size,num_sprites=4).cpu())
            t_gt_mask = torch.stack(t_gt_mask,dim=1)
            gt_masks.append(t_gt_mask.cpu())
        """
        rewards.append(reward.cpu())
        obs_encodings.append(encodings.cpu())
        actions.append(action.cpu())
        all_obs.append(obs.cpu())
        
    if save_obs:
        return torch.cat(obs_encodings,dim=0).detach().cpu(),torch.cat(actions,dim=0).detach().cpu(),torch.cat(all_obs,dim=0).detach().cpu()
    else:
        return torch.cat(obs_encodings,dim=0).detach().cpu(),torch.cat(actions,dim=0).detach().cpu(), torch.cat(rewards,dim=0).cpu()
        
def setup_models(config):
    
    config["encoder_decoder"]["in_channels"] = (3 if (config["env"]["instant_move"] or config["env"]["discrete_all_sprite_mover"]) else 9)
    print(config["encoder_decoder"]["in_channels"])
    config["dynamics"]["discrete_actions"] =  config["env"]["discrete_all_sprite_mover"]

    encoder_decoder = hydra.utils.instantiate(config["encoder_decoder"]).to(config["device"])#PatchVAE(**patch_vae).to(self.device)
    encoder_decoder.load_state_dict(torch.load(config["encoder_decoder_path"],map_location=config["device"]))
    
    config["dynamics"]["in_features"] = config["encoder_decoder"]["embed_dim"]
    config["dynamics"]["num_patches"] = encoder_decoder.num_patches
    print(f"The encoder ouputs {encoder_decoder.num_patches} patches")

    dynamics_model = hydra.utils.instantiate(config["dynamics"]).to(config["device"])
        
    return encoder_decoder, dynamics_model

@hydra.main(config_path="configs", config_name="train_dynamics")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""),("no_targets" if config["env"]["dont_show_targets"] else ""))
    dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=False,sequence_length=3)#config["env"]["discrete_all_sprite_mover"])
    num_actions = dataset[0][1].shape[1]
    config["dynamics"]["num_actions"] = num_actions
    config["dynamics"]["device"] = config["device"]
    config["dynamics"]["num_rewards"] = dataset.num_rewards
    encoder_decoder_conf = OmegaConf.load(os.path.join(os.path.dirname(config["encoder_decoder_path"]), ".hydra/config.yaml"))["encoder_decoder"]

    config["encoder_decoder"] = encoder_decoder_conf
    encoder_decoder, dynamics = setup_models(config)
    
    encoder_decoder.eval()

    for param in encoder_decoder.parameters():
        param.requires_grad = False
        
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=config["train_loop"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=128,shuffle=False,num_workers=config["train_loop"]["num_workers"])
    
    print("Creating the encodings dataset")
    with torch.no_grad():
        train_encodings_dataset = encodings_dataset(encoder_decoder, train_dataloader,save_obs=config["recons_loss"],device= config["device"])
        val_encodings_dataset = encodings_dataset(encoder_decoder, val_dataloader,save_obs=config["recons_loss"],device=config["device"])
    
    del train_dataset
    del val_dataset
    del train_dataloader
    del val_dataloader

    print("Train encodings shape",train_encodings_dataset[0].shape)
    print("Train actions shape",train_encodings_dataset[1].shape)
    print("Finished creating the encoding dataset, Training the dynamics model now")
    train_dataset = TensorDataset(*train_encodings_dataset)
    val_dataset = TensorDataset(*val_encodings_dataset)
    
    train_dataloader = DataLoader(train_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=True,num_workers=config["train_loop"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=False,num_workers=config["train_loop"]["num_workers"])
    
    optimizer = torch.optim.AdamW( list(dynamics.parameters()) , lr=config["train_loop"]["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["train_loop"]["scheduler_milestones"], gamma=0.2)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,last_epoch=-1,eta_min=1e-5)
    train(dynamics,encoder_decoder if config["recons_loss"] else None, optimizer,train_dataloader,val_dataloader,config["train_loop"]["num_epochs"],config["device"],scheduler)
    print("Training Done")
    
    
if __name__ == "__main__":

    main()
    
