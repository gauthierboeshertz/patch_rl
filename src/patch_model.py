import argparse
from unittest.mock import patch
import torch
from .utils import get_augmentation
import copy
import torch.nn.functional as F
import torch.nn as nn
import hydra
import einops
import math
from .inverse_dynamics_model import PatchInverseDynamicsModel
from collections import defaultdict


class PatchModel(nn.Module):
    def __init__(self, 
                encoder_decoder,
                inverse,
                vae_loss_weight=1,
                inverse_loss_weight=1,
                device="cpu"):
        
        super(PatchModel,self).__init__()
        
        self.device = device
        self.transforms = None
        self.vae_loss_weight = vae_loss_weight
        self.inverse_loss_weight = inverse_loss_weight
        self.encoder_decoder = encoder_decoder.to(device)
        self.inverse_model = inverse.to(device)
        
        
    def get_trainable_params(self):
        param_dict = {}
        param_dict["encoder_decoder"] = self.encoder_decoder.parameters()
        param_dict["inverse"] = self.inverse_model.parameters()
        return param_dict
    
    def save_networks(self,info=""):
        
        torch.save(self.encoder_decoder.state_dict(), f"{info}_encoder_decoder.pt")
        torch.save(self.inverse_model.state_dict(), f"{info}_inverse_model.pt")
        
    @torch.no_grad()
    def transform(self, images, transforms, augment=False):
        images = ((images.float()/255.)) 
        if augment:
            flat_images = einops.rearrange(images, "b t (r c) h w -> b (t r) c h w", r=3, c=3)
            processed_images = transforms(flat_images)
            processed_images = einops.rearrange(processed_images, "b (t r) c h w -> b t (r c) h w", r=3, c=3)
            return processed_images
        else:
            return images
        
    def remove_empty_patches_for_loss(self, prediction, target):
        #remove patches that are all zeros
        # patches is of shape (b c h w)
        empty_patches = (target.sum(axis=(1,2,3)) == 0)
        target  = target[~empty_patches]
        if isinstance(prediction,list):
            prediction = [pred[~empty_patches] for pred in prediction]
        else:
            prediction = prediction[~empty_patches]
        return prediction, target
    
    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean()
        return loss

    
    def forward(self, batch):
        """
        forward Processes a batch of transitions and returns the loss


        Args:
            batch (_type_): batch containinng tuples of (obs, action, next_obs, reward)

        Returns:
            _type_: loss to be minimized
        """        
        obs, actions, next_obs, _ = batch
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        #next_obs = next_obs.to(self.device)
        
        obs = obs.float()/255.
        
        vae_loss, encodings, vae_loss_dict = self.encoder_decoder.compute_loss_encodings(obs)
        
        
        if self.inverse_loss_weight > 0:
            inverse_loss = self.inverse_model.compute_loss(encodings, actions)
        else:
            inverse_loss = torch.tensor(0.0).to(self.device)
        
        
        loss_dict = {"vae_loss":vae_loss,"inverse_loss":inverse_loss, "recons_loss":vae_loss_dict["Reconstruction_Loss"]}
        
        return (self.vae_loss_weight*vae_loss) + (self.inverse_loss_weight*inverse_loss), loss_dict

        
