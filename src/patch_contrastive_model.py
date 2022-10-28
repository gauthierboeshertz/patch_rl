import argparse
import torch
from .networks.cnns import PatchConv2dModel
from .utils import get_augmentation, soft_update_params, maybe_transform, renormalize
import copy
from .dynamics_model import AttentionDynamicsModel
import einops
import torch.nn.functional as F
import torch.nn as nn


class PatchContrastiveModel:
    def __init__(self, 
                encoder_dict,
                dynamics_dict,
                augmentations = ["shift", "intensity"],
                target_augmentations= ["shift", "intensity"],
                val_augmentations = "none",
                aug_prob=1,
                lr=0.0001,
                target_tau=0,
                encoder_model_name="conv2d",
                device="cpu"):
        
        self.device = device
        self.aug_prob = aug_prob
        self.transforms = get_augmentation(augmentations,84)
        self.target_transforms = get_augmentation(target_augmentations, 84)
        self.val_transforms = get_augmentation(val_augmentations, 84)
        self.target_tau = target_tau #tau should be bigger than when no augmentation is used
        
        if encoder_model_name == "conv2d":
            self.encoder = PatchConv2dModel(**encoder_dict).to(self.device)

        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        
        toy_map = self.encoder( torch.zeros(1,9,128,128).to(self.device))
        num_patches = toy_map.flatten(2).shape[2]
        in_features = toy_map.shape[1]
        dynamics_dict["in_features"] = in_features
        dynamics_dict["num_patches"] = num_patches
        print(f"The encoder ouputs {num_patches} patches")
        print(f"The encoder ouputs {in_features} features per patch")

        self.forward_model = AttentionDynamicsModel(**dynamics_dict).to(self.device)
                        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),lr=lr)
        self.forward_optimizer = torch.optim.Adam(self.forward_model.parameters(),lr=lr)

        

    def project(self, x,target=False):
        if target:
            x = self.target_encoder(x)
            x = renormalize(x)
            x = x.view(x.size(0), -1)
            x = self.target_projection(x)
            return x
        else:
            x = self.encoder(x)
            x = renormalize(x)
            x = x.view(x.shape[0], -1)
            x = self.projection(x)
            return x
    
    
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
        soft_update_params(self.encoder, self.target_encoder, self.target_tau)
            
    
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

    def make_patches(self,images,target=False):
        
        if not target:
            img_features = self.encoder(images)
        else:
            img_features = self.target_encoder(images)
        
        patches = einops.rearrange(img_features, 'b c h w -> b (h w)  c')
        return patches.contiguous()
        
    def process_recursive_batch(self, batch):
    
        obs, action, _, _ = batch
        
        obs = obs.to(self.device)
        action = action.to(self.device)
        
        first_obs = self.transform(obs[:,0], self.transforms, augment=True)   
        
        patches = self.make_patches(first_obs,target=False)
        
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
                target_patches.append(self.make_patches(target_obs[:,i],target=True))
                
            target_patches = torch.cat(target_patches, dim=1)
            
        # SPR LOSS
        spr_loss = self.do_spr_loss(pred_patches, target_patches)
        
        # INVERSE LOSS
        #inverse_loss = self.do_inverse_loss(proj_latents, target_proj,action)
        
        return spr_loss

        
    def train_epoch(self,train_dataloader):
        self.encoder.train()
        self.forward_model.train()

        epoch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            
            loss = self.process_recursive_batch(batch)
            loss.backward()
            
            epoch_loss += loss.item()
            self.encoder_optimizer.step()
            self.forward_optimizer.step()
            self.update_targets()
            
        return epoch_loss/len(train_dataloader)

    def val_epoch(self,val_dataloader):
        self.encoder.eval()
        self.forward_model.eval()
        
        epoch_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                
                loss = self.process_recursive_batch(batch)
                
                epoch_loss += loss.item()
                

        return epoch_loss/len(val_dataloader)
            
    def save_models(self):
            
        torch.save(self.encoder.state_dict(), "encoder.pt")
        torch.save(self.forward_model.state_dict(), "forward_model.pt")

    
