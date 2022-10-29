import argparse
import torch
from .utils import get_augmentation, soft_update_params, maybe_transform, renormalize
import copy
from .dynamics_model import AttentionDynamicsModel
import torch.nn.functional as F
import torch.nn as nn
import hydra
from .patch_maker import PatchMaker

class PatchContrastiveModel(nn.Module):
    def __init__(self, 
                patchmaker_dict,
                dynamics_dict,
                augmentations = ["shift", "intensity"],
                target_augmentations= ["shift", "intensity"],
                val_augmentations = "none",
                aug_prob=1,
                lr=0.0001,
                target_tau=0,
                device="cpu"):
        
        super(PatchContrastiveModel,self).__init__()
        
        self.device = device
        self.aug_prob = aug_prob
        self.transforms = get_augmentation(augmentations,84)
        self.target_transforms = get_augmentation(target_augmentations, 84)
        self.val_transforms = get_augmentation(val_augmentations, 84)
        self.target_tau = target_tau #tau should be bigger than when no augmentation is used
        
        self.patchmaker = PatchMaker(**patchmaker_dict).to(self.device)
        self.target_patchmaker = copy.deepcopy(self.patchmaker).to(self.device)
        
        toy_patches = self.patchmaker(torch.zeros(1,9,128,128).to(self.device))
        num_patches = toy_patches.shape[1]
        in_features = toy_patches.shape[2]
        dynamics_dict["in_features"] = in_features
        dynamics_dict["num_patches"] = num_patches
        print(f"The encoder ouputs {num_patches} patches")
        print(f"The encoder ouputs {in_features} features per patch")

        self.forward_model = hydra.utils.instantiate(dynamics_dict).to(self.device)
                        
    
    
    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = maybe_transform(image, transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, transforms, augment=False):
        images = ((images.float()/255.)) 
        if augment:
            flat_images = images.reshape(-1, *images.shape[-3:])
            processed_images = self.apply_transforms(transforms,
                                                     flat_images)
            processed_images = processed_images.view(*images.shape[:-3],
                                                     *processed_images.shape[1:])
            return processed_images
        else:
            return images

    def update_targets(self):
        soft_update_params(self.patchmaker, self.target_patchmaker, self.target_tau)
            
    
    def do_spr_loss(self,proj_latents, targets):
        #pred_latents = self.predictor(proj_latents)
        #return self.normalized_l2_loss(pred_latents, targets)
        f_x1 = F.normalize(proj_latents, p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(targets, p=2., dim=-1, eps=1e-3)
        return F.mse_loss(f_x1, f_x2)

    def do_inverse_loss(self,proj_latents,target_proj,actions):
        obs_pairs = torch.cat([proj_latents[:,:-1], target_proj[:,1:]], dim=2)
        
        pred_actions = self.inverse_model(obs_pairs.view(obs_pairs.shape[0]*obs_pairs.shape[1],-1))
        
        pred_actions = pred_actions.view(obs_pairs.shape[0],obs_pairs.shape[1],-1)
        
        return F.mse_loss(pred_actions, actions[:,:-1],-1)
        
    def forward(self, batch):
        """
        forward Processes a batch of transitions and returns the loss


        Args:
            batch (_type_): batch containinng tuples of (obs, action, next_obs, reward)

        Returns:
            _type_: loss to be minimized
        """        
        obs, action, _, _ = batch
        
        obs = obs.to(self.device)
        action = action.to(self.device)
        
        first_obs = self.transform(obs[:,0], self.transforms, augment=True)   
        
        patches = self.patchmaker(first_obs)
        
        patches = renormalize(patches)
        
        pred_patches = [patches]
        
        for step in range(1,obs.shape[1]):
            patches,attention_weights = self.forward_model([patches, action[:,step]])
            pred_patches.append(patches)
                
        pred_patches = torch.cat(pred_patches, dim=1)
        #proj_latents = self.projection(pred_latents.view(pred_latents.shape[0]*pred_latents.shape[1], -1))
        
        with torch.no_grad():
            target_obs = self.transform(obs, self.transforms, augment=True)
            target_patches = []
            for i in range(obs.shape[1]):
                target_patches.append(self.target_patchmaker(target_obs[:,i]))
                
            target_patches = torch.cat(target_patches, dim=1)
            
        # SPR LOSS
        spr_loss = self.do_spr_loss(pred_patches, target_patches)
        
        # INVERSE LOSS
        #inverse_loss = self.do_inverse_loss(proj_latents, target_proj,action)
        
        return spr_loss

        
