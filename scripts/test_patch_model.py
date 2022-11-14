import hydra
import torch
from torch.utils.data import DataLoader
import logging
from stable_baselines3.common.utils import set_random_seed
set_random_seed(0)
from src.patch_model import PatchModel
from src.datasets import SequenceImageTransitionDataset
from hydra.utils import get_original_cwd
from collections import defaultdict
from omegaconf import  OmegaConf

logger = logging.getLogger(__name__)
import os

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
                


def test(model,  train_dataloader):
    
    _, test_loss_dict = val_epoch(model, train_dataloader)
        
    epoch_info = f"Test: "
    for k in test_loss_dict:
        epoch_info += f"{k}: {test_loss_dict[k]:.5f}, "
    #epoch_info = f"Epoch: {epoch},TRAIN : DYN Loss: {train_dyn_loss} VAE LOSS: {train_vae_loss}  INV LOSS: {train_inv_loss}||   VAL : DYN Loss: {val_dyn_loss} VAE LOSS: {val_vae_loss} INV LOSS: {val_inv_loss}"
    logger.info(epoch_info)
    print(epoch_info)
    return model

def setup_model(config):
    
    config["encoder_decoder"]["in_channels"] = (3 if (config["env"]["instant_move"] or config["env"]["discrete_all_sprite_mover"]) else 9)
    print(config["encoder_decoder"]["in_channels"])
    config["dynamics"]["discrete_actions"] =  True#config["env"]["discrete_all_sprite_mover"]

    encoder_decoder = hydra.utils.instantiate(config["encoder_decoder"]).to(config["device"])#PatchVAE(**patch_vae).to(self.device)
    encoder_decoder.load_state_dict(torch.load(config["encoder_decoder_path"],map_location=config["device"]))
    
    config["dynamics"]["in_features"] = config["encoder_decoder"]["embed_dim"]
    config["dynamics"]["num_patches"] = encoder_decoder.num_patches
    print(f"The encoder ouputs {encoder_decoder.num_patches} patches")

    dynamics_model = hydra.utils.instantiate(config["dynamics"]).to(config["device"])

    #inverse["encoder"]["in_channels"] = patch_dim 
    
    config["inverse"]["encoder"]["in_channels"] = config["encoder_decoder"]["embed_dim"] * 2
    config["inverse"]["mlp"]["input_size"] = config["inverse"]["encoder"]["channels"][-1]*encoder_decoder.num_patches
    config["inverse"]["mlp"]["output_size"] = config["dynamics"]["num_actions"] * config["dynamics"]["action_dim"]
    
    inverse_model = hydra.utils.instantiate(config["inverse"]).to(config["device"])
    
    model = PatchModel(encoder_decoder=encoder_decoder,dynamics=dynamics_model,inverse=inverse_model,
                       dyn_loss_weight=config["loss_weights"]["dyn_loss_weight"],
                       vae_loss_weight=config["loss_weights"]["vae_loss_weight"],
                       inverse_loss_weight=config["loss_weights"]["inverse_loss_weight"],
                       device=config["device"])
    return model

@hydra.main(config_path="configs", config_name="test_patch_model")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""))
    test_dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=False,sequence_length=2)#config["env"]["discrete_all_sprite_mover"])
    num_actions = test_dataset[0][1].shape[1]
    config["dynamics"]["num_actions"] = num_actions
    
    encoder_decoder_conf = OmegaConf.load(os.path.join(os.path.dirname(config["encoder_decoder_path"]), ".hydra/config.yaml"))["encoder_decoder"]
    config["encoder_decoder"] = encoder_decoder_conf
    model = setup_model(config)
    
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True,num_workers=config["num_workers"])
    
    test(model,test_dataloader)
    print("Training Done")
    
    
if __name__ == "__main__":

    main()
    
