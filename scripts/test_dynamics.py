import hydra
import numpy as np
import torch
import logging

from stable_baselines3.common.utils import set_random_seed
set_random_seed(0)
from omegaconf import OmegaConf
from progress.bar import Bar
import os
from hydra.utils import get_original_cwd, to_absolute_path
from collections import defaultdict
import einops
from src.datasets import SequenceImageTransitionDataset
from src.patch_utils import patches_to_image
logger = logging.getLogger(__name__)
from src.mask_utils import make_gt_causal_mask, get_cc_dicts_from_mask
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt    
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_fscore_support
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_recons_loss(obs, action, next_obs,dynamics, encoder_decoder,):
    t_obs = einops.rearrange(obs, "b t c h w -> (b t) c h w")
    encodings = encoder_decoder.get_encoding_for_dynamics(t_obs/255.)
    encodings = einops.rearrange(encodings, "(b t n) c -> (b t) n c", b = action.shape[0],t=action.shape[1])
    t_action = einops.rearrange(action, "b t a -> (b t) a")
    dynamics_encodings = dynamics([encodings,t_action])[0]
    dynamics_encodings = einops.rearrange(dynamics_encodings, "(b t) n c -> (b t n) c",b=action.shape[0],t=action.shape[1])
    dyn_recons = encoder_decoder.decode(dynamics_encodings)
    dyn_recons = einops.rearrange(dyn_recons, "(b t n) c h w -> b t n c h w", b=obs.shape[0],t=action.shape[1])
    
    dyn_recons_images = []
    for t in range(action.shape[1]):
        dyn_recons_images.append(patches_to_image(dyn_recons[:,t],patch_size=16,image_size=obs.shape[3]))
    
    dyn_recons_image = torch.stack(dyn_recons_images,dim=1) 
    
    dyn_recons_loss = torch.mean((dyn_recons_image - (next_obs/255))**2)
    
    return dyn_recons_loss


def compare_cc(pred_ccs,gt_ccs):
    
    pred_action_ccs = [cc for cc in pred_ccs if len(cc["actions"]) > 0]
    gt_action_ccs = [cc for cc in gt_ccs if len(cc["actions"]) > 0]
    
    cc_metrics = {}
    
    for i in range(1,4):
        cc_metrics[f"num_ccs_with_{i}_actions"] = sum([len(cc["actions"]) == i for cc in gt_action_ccs])

    cc_metrics["abs_len_diff_all_ccs"] = abs(len(pred_ccs) - len(gt_ccs))
    cc_metrics["abs_len_diff_action_ccs"] = abs(len(pred_action_ccs) - len(gt_action_ccs))
    
    cc_metrics["cc_num_patches_diff"] = defaultdict(list)
    for gt_action_cc in gt_action_ccs:
        same_action_ccs = [pred_action_cc for pred_action_cc in pred_action_ccs if pred_action_cc["actions"] == gt_action_cc["actions"]]
        if len(same_action_ccs) != 1:
            continue
        same_action_cc = same_action_ccs[0]
        cc_metrics["cc_num_patches_diff"][len(gt_action_cc["actions"])].append(abs(len(same_action_cc["patches"]) - len(gt_action_cc["patches"])))

        
    return cc_metrics
        
        
def compute_cc_metrics(pred_masks,gt_masks,num_actions,num_patches):

    cc_metrics_dicts = []
    for b in range(pred_masks.shape[0]):
        gt_ccs = get_cc_dicts_from_mask(gt_masks[b],num_actions=num_actions,num_patches=num_patches)
        pred_ccs = get_cc_dicts_from_mask(pred_masks[b],num_actions=num_actions,num_patches=num_patches)
        cc_metrics_dicts.append(compare_cc(pred_ccs,gt_ccs))
    
    cc_metrics = {}
    
    possible_num_patches_diff_keys = [i for i in range(num_actions)]
    for key in possible_num_patches_diff_keys:
        all_num_patches_diff = [cc_m["cc_num_patches_diff"][key] for cc_m in cc_metrics_dicts if key in cc_m["cc_num_patches_diff"].keys()]
        all_num_patches_diff = [item for sublist in all_num_patches_diff for item in sublist] ##flatten list
        if len(all_num_patches_diff) > 0:
            cc_metrics[f"cc_num_patches_diff_{key}_stats"] = pd.DataFrame(all_num_patches_diff).describe()
    
    for i in range(1,4):
        cc_metrics[f"num_ccs_with_{i}_actions"] = sum([cc_m[f"num_ccs_with_{i}_actions"] for cc_m in cc_metrics_dicts])
        
    abs_len_diff_all_ccs = np.array([cc_m["abs_len_diff_all_ccs"] for cc_m in cc_metrics_dicts])
    abs_len_diff_action_ccs = np.array([cc_m["abs_len_diff_action_ccs"] for cc_m in cc_metrics_dicts])
    
    abs_len_diff_all_ccs_stats = pd.DataFrame(abs_len_diff_all_ccs).describe()
    abs_len_diff_action_ccs_stats = pd.DataFrame(abs_len_diff_action_ccs).describe()

    cc_metrics["abs_len_diff_all_ccs_stats"] = abs_len_diff_all_ccs_stats
    cc_metrics["abs_len_diff_action_ccs_stats"] = abs_len_diff_action_ccs_stats
    
    return cc_metrics


def compute_mask_metrics(pred_masks_prob,gt_masks,threshold=0.2):
    
    pred_masks_prob = (pred_masks_prob > threshold).astype(int)
    print(pred_masks_prob.shape)
    print(np.eye(pred_masks_prob.shape[-1])[np.newaxis, ...].shape)
    
    remove_self_attention = np.eye(pred_masks_prob.shape[-1])[np.newaxis, ...]
    remove_self_attention = np.repeat(remove_self_attention,pred_masks_prob.shape[0],axis=0)
    remove_self_attention[pred_masks_prob == 0] = 0 # dont remove the values associated to the actions
                                                    # but remove the values associated to self atttention 
                                                    # because they are not predicted
                                                    
    pred_masks_prob = pred_masks_prob - remove_self_attention#np.eye(pred_masks_prob.shape[-1])[np.newaxis, ...]
    print(pred_masks_prob.shape)
    gt_masks = gt_masks - remove_self_attention
    gt_masks_flat = gt_masks.flatten()
    pred_masks_flat = pred_masks_prob.flatten()

    conf_mat = ConfusionMatrixDisplay.from_predictions(gt_masks_flat, pred_masks_flat)
    conf_mat.figure_.savefig("conf_mat.png")
    all_metrics = precision_recall_fscore_support(gt_masks_flat, pred_masks_flat)
    print(all_metrics)
    roc_curve = RocCurveDisplay.from_predictions(gt_masks_flat, pred_masks_flat)
    roc_curve.figure_.savefig("roc_curve.png")
    
    metrics = {}
    metrics["precision"] = all_metrics[0][1]
    metrics["recall"] = all_metrics[1][1]
    metrics["f1"] = all_metrics[2][1]
    
    return metrics

    
def compute_mask_cc_metrics(pred_masks_probs,gt_masks,threshold=0.2,num_actions=4,num_patches=16):

    pred_masks = pred_masks_probs > threshold

    
    cc_metrics = compute_cc_metrics(pred_masks,gt_masks,num_actions,num_patches)
    mask_metrics = compute_mask_metrics(pred_masks,gt_masks,threshold=threshold)

    all_metrics = {}
    all_metrics.update(mask_metrics)
    all_metrics.update(cc_metrics)
    return all_metrics


def get_masks_probs_and_gt(obs,action,next_obs,dynamics ,encoder_decoder):
    t_obs = einops.rearrange(obs, "b t c h w -> (b t) c h w")
    t_action = einops.rearrange(action, "b t a -> (b t) a")
    t_next_obs = einops.rearrange(next_obs, "b t c h w -> (b t) c h w")
    
    images = torch.stack([t_obs,t_next_obs],dim=1)
    gt_causal_mask = make_gt_causal_mask(t_obs.cpu(),action.cpu(),t_next_obs.cpu(),patch_size=encoder_decoder.patch_size,num_sprites=dynamics.action_embedding.num_actions)
    pred_causal_masks_probs, _ = dynamics.get_causal_mask(t_obs/255., t_action,encoder=encoder_decoder)

    return pred_causal_masks_probs.cpu(), gt_causal_mask.cpu()


def val_epoch(dynamics,encoder_decoder,dataloader):
    dynamics.eval()
    encoder_decoder.eval()
    epoch_loss_dict = defaultdict(float)
    total_loss = 0
    
    gt_masks = []
    pred_masks_probs = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            obs = batch[0].to(device)
            action = batch[1].to(device)
            next_obs = batch[2].to(device)
            
            epoch_loss_dict["dyn_recons_loss"] += compute_recons_loss(obs,action,next_obs,dynamics=dynamics,encoder_decoder=encoder_decoder).item()         
            
            batch_pred_masks_probs, batch_gt_masks = get_masks_probs_and_gt(obs,action,next_obs,dynamics ,encoder_decoder)
            gt_masks.append(batch_gt_masks)
            pred_masks_probs.append(batch_pred_masks_probs)
            
        
    epoch_loss_dict["dyn_recons_loss"] /= len(dataloader)
        
    ## compute mask related metrics
    gt_masks = torch.cat(gt_masks,dim=0).cpu().numpy()
    pred_masks_probs = torch.cat(pred_masks_probs,dim=0).cpu().numpy()
    mask_cc_metrics = compute_mask_cc_metrics(pred_masks_probs,gt_masks,threshold=dynamics.causal_mask_threshold,num_actions=dynamics.action_embedding.num_actions,num_patches=encoder_decoder.num_patches)

    epoch_loss_dict.update(mask_cc_metrics)
    
    return  total_loss/len(dataloader), epoch_loss_dict
                

def test(dynamics, encoder_decoder, train_dataloader):
    
    _, test_loss_dict = val_epoch(dynamics,encoder_decoder, train_dataloader)
        
    epoch_info = f"Test: "
    for k in test_loss_dict:
        if isinstance(test_loss_dict[k],(float)):
            epoch_info += f"{k}: {test_loss_dict[k]:.5f} \n "
        elif isinstance(test_loss_dict[k],(pd.DataFrame, pd.Series)):
            df_string = test_loss_dict[k].to_string().replace('\n', '\n\t')
            epoch_info += f"{k}: {df_string} \n "
        else:
            epoch_info += f"{k}: {test_loss_dict[k]} \n "
    #epoch_info = f"Epoch: {epoch},TRAIN : DYN Loss: {train_dyn_loss} VAE LOSS: {train_vae_loss}  INV LOSS: {train_inv_loss}||   VAL : DYN Loss: {val_dyn_loss} VAE LOSS: {val_vae_loss} INV LOSS: {val_inv_loss}"
    logger.info(epoch_info)


def setup_models(config):
    
    
    config["encoder_decoder"]["in_channels"] = (3 if (config["env"]["instant_move"] or config["env"]["discrete_all_sprite_mover"]) else 9)
    print(config["encoder_decoder"]["in_channels"])
    config["dynamics"]["discrete_actions"] =  config["env"]["discrete_all_sprite_mover"]

    encoder_decoder = hydra.utils.instantiate(config["encoder_decoder"]).to(device)#PatchVAE(**patch_vae).to(self.device)
    encoder_decoder.load_state_dict(torch.load(config["encoder_decoder_path"],map_location=device))
    
    config["dynamics"]["in_features"] = config["encoder_decoder"]["embed_dim"]
    config["dynamics"]["num_patches"] = encoder_decoder.num_patches
    print(f"The encoder ouputs {encoder_decoder.num_patches} patches")
    dynamics_model = hydra.utils.instantiate(config["dynamics"]).to(device)
    print(torch.load(config["dynamics_model_path"],map_location=device))
    dynamics_model.load_state_dict(torch.load(config["dynamics_model_path"],map_location=device))
    
    encoder_decoder.eval()
    dynamics_model.eval()
    return encoder_decoder, dynamics_model

@hydra.main(config_path="configs", config_name="test_dynamics")
def main(config):
    
    print("Training with config: {}".format(config))
    
    if os.path.isfile(config["test_data_path" ]):
        test_dataset = SequenceImageTransitionDataset(data_path=config["test_data_path"],onehot_action=False,sequence_length=1)#config["env"]["discrete_all_sprite_mover"])
    else:
        data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""),("no_targets" if config["env"]["dont_show_targets"] else ""))
        test_dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=False,sequence_length=1)#config["env"]["discrete_all_sprite_mover"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    num_actions = test_dataset[0][1].shape[1]
    trained_dynamics_conf = OmegaConf.load(os.path.join(os.path.dirname(config["dynamics_model_path"]), ".hydra/config.yaml"))
    config["dynamics"] = trained_dynamics_conf["dynamics"]
    config["dynamics"]["num_actions"] = num_actions
    config["dynamics"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["dynamics"]["num_rewards"] = 3#test_dataset.num_rewards
    config["encoder_decoder_path"] = trained_dynamics_conf["encoder_decoder_path"]
    print("Loading encoder decoder from {}".format(config["encoder_decoder_path"]))
    encoder_decoder_conf = OmegaConf.load(os.path.join(os.path.dirname(config["encoder_decoder_path"]), ".hydra/config.yaml"))["encoder_decoder"]
    config["encoder_decoder"] = encoder_decoder_conf

    encoder_decoder, dynamics = setup_models(config)
    test(dynamics, encoder_decoder,test_dataloader)
    
    
if __name__ == "__main__":

    main()

    