import argparse
import torch
from .utils import get_augmentation, soft_update_params, renormalize
import copy
import torch.nn.functional as F
import torch.nn as nn
import hydra
from .patchmaker import PatchMaker
from .patch_decoder import PatchDecoder
import einops
from .networks.cnns import MLP

class PatchContrastiveModel(nn.Module):
    def __init__(self, 
                patchmaker,
                dynamics,
                patch_decoder,
                augmentations = ["blur"],
                target_augmentations= ["blur", "intensity"],
                val_augmentations = "none",
                aug_prob=1,
                target_tau=0,
                dyn_loss_weight=1,
                byol_loss_weight=1,
                device="cpu"):
        
        super(PatchContrastiveModel,self).__init__()
        
        self.device = device
        self.aug_prob = aug_prob
        self.transforms = get_augmentation(augmentations,128,aug_prob)
        self.target_transforms = get_augmentation(target_augmentations, 128,aug_prob)
        self.val_transforms = get_augmentation(val_augmentations, 128,aug_prob)
        self.target_tau = target_tau #tau should be bigger than when no augmentation is used
        self.dyn_loss_weight = dyn_loss_weight
        self.byol_loss_weight = byol_loss_weight
        self.patchmaker = PatchMaker(**patchmaker).to(self.device)
        self.target_patchmaker = copy.deepcopy(self.patchmaker).to(self.device)
        
        toy_patches = self.patchmaker(torch.zeros(1,9,128,128).to(self.device))
        num_patches = toy_patches.shape[1]
        patch_dim = toy_patches.shape[2]
        dynamics["in_features"] = patch_dim
        dynamics["num_patches"] = num_patches
        print(f"The encoder ouputs {num_patches} patches")
        print(f"The encoder ouputs {patch_dim} features per patch")

        self.forward_model = hydra.utils.instantiate(dynamics).to(self.device)
        
        #byol models
        self.predictor = MLP(input_size=patch_dim, hidden_size=4096,output_size=patch_dim,num_layers= 3).to(self.device)
        
        self.projector = MLP(input_size=patch_dim, hidden_size=4096,output_size=patch_dim,num_layers= 3).to(self.device)
        self.target_projector = copy.deepcopy(self.projector).to(self.device)
        
        self.patch_decoder = PatchDecoder(**patch_decoder).to(self.device)
        
        
    def get_trainable_params(self):
        param_dict = {}
        param_dict["encoder"] = self.patchmaker.parameters()
        param_dict["dynamics"] = self.forward_model.parameters()
        param_dict["predictor"] = self.predictor.parameters()
        param_dict["projector"] = self.projector.parameters()
        param_dict["patch_decoder"] = self.patch_decoder.parameters()

        return param_dict
    
    def save_networks(self):
        torch.save(self.patchmaker.state_dict(), "patchmaker.pt")
        torch.save(self.forward_model.state_dict(), "dynamics.pt")
        torch.save(self.patch_decoder.state_dict(), "patch_decoder.pt")

        
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

    def update_targets(self):
        soft_update_params(self.patchmaker, self.target_patchmaker, self.target_tau)
        soft_update_params(self.projector, self.target_projector, self.target_tau)
            
    
    def normalized_ctr_loss(self,proj_latents, targets):
        #pred_latents = self.predictor(proj_latents)
        #return self.normalized_l2_loss(pred_latents, targets)
        f_x1 = F.normalize(proj_latents, p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(targets, p=2., dim=-1, eps=1e-3)
        return F.mse_loss(f_x1, f_x2)

    
    
    def do_byol_loss(self, obs):
        def loss_fn(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x * y).sum(dim=-1)
        
        def byol_loss_one_obs(t_obs_1, t_obs_2):
            patches_1 = self.patchmaker(t_obs_1)
            patches_2 = self.patchmaker(t_obs_2)

            proj_patches_1 = self.projector(patches_1)
            proj_patches_2 = self.projector(patches_2)
            
            pred_1 = self.predictor(proj_patches_1)
            pred_2 = self.predictor(proj_patches_2)

            with torch.no_grad():
                target_patches_1 = self.target_patchmaker(t_obs_1)
                target_patches_2 = self.target_patchmaker(t_obs_2)
                target_proj_patches_1 = self.projector(target_patches_1)
                target_proj_patches_2 = self.projector(target_patches_2)

            loss_one = loss_fn(pred_1, target_proj_patches_2.detach())
            loss_two = loss_fn(pred_2, target_proj_patches_1.detach())

            loss = loss_one + loss_two
            return loss.mean()

        transformed_obs_1 = self.transform(obs, self.transforms, augment=True)   
        transformed_obs_2 = self.transform(obs, self.transforms, augment=True)
        
        total_loss = 0
        for i in range(obs.shape[1]):
            total_loss += byol_loss_one_obs(transformed_obs_1[:,i], transformed_obs_2[:,i])
        return total_loss/obs.shape[1]

            
    def do_dynamic_loss(self,obs,actions,next_obs):
        transformed_obs = self.transform(obs, self.transforms, augment=False)   
        transformed_next_obs = self.transform(next_obs, self.transforms, augment=False)
        first_obs_patch = self.patchmaker(transformed_obs[:,0])

        #use the first patch and use dynamics to predict the next patch recursively
        forward_patches = [first_obs_patch]
        for step in range(1,obs.shape[1]):
            forward_patch,_ = self.forward_model([forward_patches[step-1], actions[:,step-1]])
            forward_patches.append(forward_patch)
        forward_patches = torch.stack(forward_patches, dim=1)
        forward_patches = forward_patches[:,1:]
        
        decoded_patches = []
        for step in range(forward_patches.shape[1]):
            decoded_patches.append(self.patch_decoder(forward_patches[:,step]))
        decoded_patches = torch.stack(decoded_patches, dim=1).contiguous()
        
        next_image_patches = []
        for step in range(1,transformed_next_obs.shape[1]):
            next_image_patches.append(einops.rearrange(transformed_next_obs[:,step], "b c (h p1) (w p2) -> b (h w) c p1 p2", p1=self.patchmaker.patch_size, p2=self.patchmaker.patch_size))
        next_image_patches = torch.stack(next_image_patches, dim=1).contiguous()
        return F.mse_loss(decoded_patches, next_image_patches) 
    
    
    
    
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
                
        # dyn_loss
        dyn_loss = self.do_dynamic_loss(obs,actions,next_obs)
        
        #byol loss
        if self.byol_loss_weight <= 0:
            byol_loss = torch.tensor(0.0).to(self.device)
        else:
            byol_loss = self.do_byol_loss(obs)
        
        return (self.dyn_loss_weight*dyn_loss) + (self.byol_loss_weight * byol_loss), dyn_loss, byol_loss 

        
