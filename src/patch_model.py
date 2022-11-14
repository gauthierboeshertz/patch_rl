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
                dynamics,
                inverse,
                dyn_loss_weight=1,
                vae_loss_weight=1,
                inverse_loss_weight=1,
                device="cpu"):
        
        super(PatchModel,self).__init__()
        
        self.device = device
        self.transforms = None
        self.dyn_loss_weight = dyn_loss_weight
        self.vae_loss_weight = vae_loss_weight
        self.inverse_loss_weight = inverse_loss_weight
        self.encoder_decoder = encoder_decoder.to(device)
        self.dynamics_model = dynamics.to(device)
        self.inverse_model = inverse.to(device)
        
        
    def get_trainable_params(self):
        param_dict = {}
        param_dict["encoder_decoder"] = self.encoder_decoder.parameters()
        param_dict["dynamics"] = self.dynamics_model.parameters()
        param_dict["inverse"] = self.inverse_model.parameters()
        return param_dict
    
    def save_networks(self,info=""):
        torch.save(self.dynamics_model.state_dict(), f"{info}_dynamics.pt")
        torch.save(self.encoder_decoder.state_dict(), f"{info}_encoder_decoder.pt")

        
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

    def do_inverse_loss(self, encodings, actions):
        num_patches_sqrt = int(math.sqrt(encodings.shape[-2]))
        
        encodings = einops.rearrange(encodings, "b t (h w) c -> b t c h w", h=num_patches_sqrt, w=num_patches_sqrt)
        obs_pairs = torch.stack([encodings[:,:-1], encodings[:,1:]], dim=2)
        obs_pairs = einops.rearrange(obs_pairs, "b t p c h w -> (b t) p c h w", h=num_patches_sqrt, w=num_patches_sqrt)
        pred_actions = self.inverse_model(obs_pairs)
        pred_actions = einops.rearrange(pred_actions, "b (a h) -> b a h", a=self.dynamics_model.action_embedding.num_actions)
        actions = actions[:,:-1]
        actions = einops.rearrange(actions, "b t a -> (b t) a ").long()
        #actions = actions.permute(0,2,1).squeeze().long()
        #actions = actions#.max(dim=-1)
        
        return F.cross_entropy(pred_actions, actions)
    
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
        
        obs = self.transform(obs, self.transforms, augment=False)
        
        vae_loss, encodings, vae_loss_dict = self.encoder_decoder.compute_loss_encodings(obs)
        
        if self.dyn_loss_weight > 0:
            #encodings = encodings[:,0]
            del obs
            dyn_loss, _ = self.dynamics_model.compute_loss(encodings,actions)
        else:
            dyn_loss = torch.tensor(0.0).to(self.device)
        
        if self.inverse_loss_weight > 0:
            inverse_loss = self.do_inverse_loss(encodings, actions)
        else:
            inverse_loss = torch.tensor(0.0).to(self.device)
        
        
        loss_dict = {"vae_loss":vae_loss, "dyn_loss":dyn_loss, "inverse_loss":inverse_loss, "recons_loss":vae_loss_dict["Reconstruction_Loss"]}
        
        return (self.vae_loss_weight*vae_loss) + (self.dyn_loss_weight*dyn_loss) + (self.inverse_loss_weight*inverse_loss), loss_dict

        
