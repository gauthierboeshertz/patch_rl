import hydra
import numpy as np
import torch
import logging

from stable_baselines3.common.utils import set_random_seed
set_random_seed(0)
from omegaconf import OmegaConf
from progress.bar import Bar
import os
from src.patch_model import PatchModel
from hydra.utils import get_original_cwd, to_absolute_path
from collections import defaultdict
import einops
from src.datasets import SequenceImageTransitionDataset
from src.patch_utils import patches_to_image
logger = logging.getLogger(__name__)

    
    
def val_epoch(dynamics,encoder_decoder,dataloader,device):
    dynamics.eval()
    encoder_decoder.eval()
    epoch_loss_dict = defaultdict(float)
    total_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            obs = batch[0].to(device)
            action = batch[1].to(device)
            next_obs = batch[2].to(device)
            t_obs = einops.rearrange(obs, "b t c h w -> (b t) c h w")
            encodings = encoder_decoder.get_encoding_for_dynamics(t_obs)
            encodings = einops.rearrange(encodings, "(b t n) c -> (b t) n c", b = action.shape[0],t=action.shape[1])
            t_action = einops.rearrange(action, "b t a -> (b t) a")
            dynamics_encodings, _ = dynamics([encodings,t_action])
            dynamics_encodings = einops.rearrange(dynamics_encodings, "(b t) n c -> (b t n) c",b=action.shape[0],t=action.shape[1])
            dyn_recons = encoder_decoder.decode(dynamics_encodings)
            dyn_recons = einops.rearrange(dyn_recons, "(b t n) c h w -> b t n c h w", b=obs.shape[0],t=action.shape[1])
            
            dyn_recons_images = []
            for t in range(action.shape[1]):
                dyn_recons_images.append(patches_to_image(dyn_recons[:,t],patch_size=16,image_size=obs.shape[3]))
            
            dyn_recons_image = torch.stack(dyn_recons_images,dim=1) 
            dyn_recons_loss = torch.mean((dyn_recons_image - (next_obs/255))**2)

            epoch_loss_dict["dyn_recons_loss"] += dyn_recons_loss.item()         
            
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= len(dataloader)

    return  total_loss/len(dataloader), epoch_loss_dict
                


def test(dynamics, encoder_decoder, train_dataloader,device):
    
    _, test_loss_dict = val_epoch(dynamics,encoder_decoder, train_dataloader,device)
        
    epoch_info = f"Test: "
    for k in test_loss_dict:
        epoch_info += f"{k}: {test_loss_dict[k]:.5f}, "
    #epoch_info = f"Epoch: {epoch},TRAIN : DYN Loss: {train_dyn_loss} VAE LOSS: {train_vae_loss}  INV LOSS: {train_inv_loss}||   VAL : DYN Loss: {val_dyn_loss} VAE LOSS: {val_vae_loss} INV LOSS: {val_inv_loss}"
    logger.info(epoch_info)
    print(epoch_info)
    



def setup_models(config):
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""))
    dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=False,sequence_length=2)#config["env"]["discrete_all_sprite_mover"])
    num_actions = dataset[0][1].shape[1]
    config["dynamics"]["num_actions"] = num_actions
    
    config["encoder_decoder"]["in_channels"] = (3 if (config["env"]["instant_move"] or config["env"]["discrete_all_sprite_mover"]) else 9)
    print(config["encoder_decoder"]["in_channels"])
    config["dynamics"]["discrete_actions"] =  True

    encoder_decoder = hydra.utils.instantiate(config["encoder_decoder"]).to(config["device"])#PatchVAE(**patch_vae).to(self.device)
    encoder_decoder.load_state_dict(torch.load(config["encoder_decoder_path"],map_location=config["device"]))
    
    config["dynamics"]["in_features"] = config["encoder_decoder"]["embed_dim"]
    config["dynamics"]["num_patches"] = encoder_decoder.num_patches
    print(f"The encoder ouputs {encoder_decoder.num_patches} patches")
    dynamics_model = hydra.utils.instantiate(config["dynamics"]).to(config["device"])
    dynamics_model.load_state_dict(torch.load(config["dynamics_model_path"],map_location=config["device"]))
    return encoder_decoder, dynamics_model

@hydra.main(config_path="configs", config_name="test_dynamics")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""))
    test_dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=False,sequence_length=2)#config["env"]["discrete_all_sprite_mover"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    num_actions = test_dataset[0][1].shape[1]
    config["dynamics"]["num_actions"] = num_actions

    trained_dynamics_conf = OmegaConf.load(os.path.join(os.path.dirname(config["dynamics_model_path"]), ".hydra/config.yaml"))
    config["dynamics"] = trained_dynamics_conf["dynamics"]

    config["encoder_decoder_path"] = trained_dynamics_conf["encoder_decoder_path"]
    print("Loading encoder decoder from {}".format(config["encoder_decoder_path"]))
    encoder_decoder_conf = OmegaConf.load(os.path.join(os.path.dirname(config["encoder_decoder_path"]), ".hydra/config.yaml"))["encoder_decoder"]
    config["encoder_decoder"] = encoder_decoder_conf

    encoder_decoder, dynamics = setup_models(config)
    test(dynamics, encoder_decoder,test_dataloader,config["device"])
    
    
if __name__ == "__main__":

    main()
    
