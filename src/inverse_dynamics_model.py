import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import hydra
import einops
import math
from einops.layers.torch import Rearrange

class PatchInverseDynamicsModel(torch.nn.Module):
    """
    Predict action from the observation and the next observation.
    """

    def __init__(
            self,
            encoder_dict,
            mlp_head_dict,
            num_patches
            ):
        super(PatchInverseDynamicsModel,self).__init__()
        
        self.encoder = hydra.utils.instantiate(encoder_dict)
        self.mlp_head = hydra.utils.instantiate(mlp_head_dict)
        self.sqrt_num_patches = math.sqrt(num_patches)
        self.patches_to_embed = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=self.sqrt_num_patches, w=self.sqrt_num_patches),
                                    self.encoder)        
    def forward(self, input):
        """ Predicts the action from the observation and the next observation with patch embeddings
        """
        obs_patches = input[0]
        next_obs_patches = input[1]
        
        obs_embed = self.patches_to_embed(obs_patches)
        next_obs_embed = self.patches_to_embed(next_obs_patches)
        
        merged_embed = torch.cat([obs_embed, next_obs_embed], dim=1)
        
        return self.mlp_head(merged_embed)                
        
