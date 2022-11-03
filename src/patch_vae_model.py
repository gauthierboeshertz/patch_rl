import argparse
from unittest.mock import patch
import torch
from .utils import get_augmentation
import copy
import torch.nn.functional as F
import torch.nn as nn
import hydra
import einops
from .patch_vae import PatchVAE

class PatchVAEModel(nn.Module):
    def __init__(self, 
                patch_vae,
                dynamics,
                augmentations = ["blur"],
                aug_prob=1,
                dyn_loss_weight=1,
                dyn_regularizer_weight=1,
                vae_loss_weight=1,
                device="cpu"):
        
        super(PatchVAEModel,self).__init__()
        
        self.device = device
        self.aug_prob = aug_prob
        self.transforms = get_augmentation(augmentations,128,aug_prob)
        self.dyn_loss_weight = dyn_loss_weight
        self.dyn_regularizer_weight = dyn_regularizer_weight
        self.vae_loss_weight = vae_loss_weight
        
        print(patch_vae)
        print(dynamics)
        self.patch_vae = hydra.utils.instantiate(patch_vae).to(self.device)#PatchVAE(**patch_vae).to(self.device)
        self.patch_size = patch_vae.patch_size
        toy_patches = self.patch_vae(torch.zeros(1,patch_vae.in_channels,128,128).to(self.device))[1]
        toy_patches = einops.rearrange(toy_patches, "(b n) c  -> b n c", b=1)
        num_patches = toy_patches.shape[1]
        patch_dim = toy_patches.shape[2]
        dynamics["in_features"] = patch_dim
        dynamics["num_patches"] = num_patches
        print(f"The encoder ouputs {num_patches} patches")
        print(f"The encoder ouputs {patch_dim} features per patch")

        self.forward_model = hydra.utils.instantiate(dynamics).to(self.device)
        
        
        
    def get_trainable_params(self):
        param_dict = {}
        param_dict["patch_vae"] = self.patch_vae.parameters()
        param_dict["dynamics"] = self.forward_model.parameters()

        return param_dict
    
    def save_networks(self,info=""):
        torch.save(self.forward_model.state_dict(), f"{info}_dynamics.pt")
        torch.save(self.patch_vae.state_dict(), f"{info}_patch_vae.pt")

        
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
    
    def do_vae_loss(self,obs):
        
        transformed_obs = self.transform(obs, self.transforms, augment=False)  
        recons, z, mu, log_var = self.patch_vae(transformed_obs)
        patches = einops.rearrange(transformed_obs, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=self.patch_size, p2=self.patch_size)
        (recons, mu_removed, log_var), patches = self.remove_empty_patches_for_loss([recons, mu, log_var], patches) 
        loss_dict = self.patch_vae.loss_function(recons, patches, mu_removed, log_var)
                #loss_dict[k] = 0
        return loss_dict["loss"], z
    
    def do_dynamic_loss(self,encoding,actions,next_obs):
        #transformed_next_obs = self.transform(next_obs, self.transforms, augment=False)

        #use the first patch and use dynamics to predict the next patch recursively
        dynamic_input = encoding[:,0,:]
        forwards_pred = []
        attention_weights = []
        for step in range(1,actions.shape[1]):
            forward_encodings, attn_weight = self.forward_model([dynamic_input, actions[:,step-1]])
            dynamic_input = forward_encodings
            forwards_pred.append(forward_encodings)
            attention_weights.append(attn_weight)
        forwards_pred = torch.stack(forwards_pred, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        
        if self.dyn_regularizer_weight > 0:
            regularizer_loss = (attention_weights.diagonal(dim1=-2,dim2=-1)).pow(2).sum(dim=1).mean()
        else:
            regularizer_loss = actions.new_zeros(1)
        #decoded_patches = []
        #for step in range(forward_patches.shape[1]):
        #    fp_to_decode = einops.rearrange(forward_patches[:,step], "b n c -> (b n) c ")
        #    fp_decoded = self.patch_vae.decode(fp_to_decode)
        #    #decoded_patches.append(einops.rearrange(fp_decoded, "(b n) c h w  -> b n c h w",b=next_obs.shape[0]))
        #    decoded_patches.append(fp_decoded)
            
        #decoded_patches = torch.stack(decoded_patches, dim=1).contiguous()#.cpu()
        #decoded_patches = einops.rearrange(decoded_patches, "b t c h w -> (b t) c h w")
        
        #next_image_patches = []
        #for step in range(0,transformed_next_obs.shape[1]):
        #    next_image_patches.append(einops.rearrange(transformed_next_obs[:,step], "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=self.patch_vae.patch_size, p2=self.patch_vae.patch_size))
        #next_image_patches = torch.stack(next_image_patches, dim=1).contiguous()#.to(self.device)
        #next_image_patches = einops.rearrange(next_image_patches, "b t c h w -> (b t) c h w")
        #decoded_patches, next_image_patches = self.remove_empty_patches_for_loss(decoded_patches, next_image_patches) 
        return F.mse_loss(forwards_pred, encoding[:,1:]) +  self.dyn_regularizer_weight*regularizer_loss   #self.normalized_ctr_loss(forward_patches, target_patches)
    
    
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
        next_obs = next_obs.to(self.device)
        
        vae_loss = 0
        
        #for i in range(obs.shape[1]):
        #    if i == 0:
        #        vae_loss, encodings = self.do_vae_loss(obs[:,i])
        #    else:
        #        vae_loss += self.do_vae_loss(obs[:,i])[0]
        vae_loss, encodings = self.do_vae_loss(einops.rearrange(obs, "b t c h w -> (b t) c h w"))
        # dyn_loss
        if self.dyn_loss_weight > 0:
            encodings = einops.rearrange(encodings, "(b t n) c -> b t n c", b=obs.shape[0],t=obs.shape[1]).clone().detach()
            #encodings = encodings[:,0]
            del obs
            dyn_loss = self.do_dynamic_loss(encodings,actions,next_obs)
        else:
            dyn_loss = torch.tensor(0.0).to(self.device)
            
        return (self.vae_loss_weight*vae_loss) + (self.dyn_loss_weight*dyn_loss), vae_loss, dyn_loss

        
