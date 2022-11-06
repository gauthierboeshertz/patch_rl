import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import hydra
import einops
import math
from .networks.cnns import Conv2dModel, MLP

class PatchInverseDynamicsModel(torch.nn.Module):
    """
    Predict action from the observation and the next observation.
    """

    def __init__(
            self,
            encoder,
            mlp,
            ):
        super(PatchInverseDynamicsModel,self).__init__()
        
        print(encoder)
        self.encoder = Conv2dModel(**encoder)
        self.mlp_head = MLP(**mlp)

        
    def forward(self, input):
        """ Predicts the action from the observation and the next observation with patch embeddings
        """
        assert len(input.shape) == 5, "The input should be of shape (batch_size,2, patch_dim, height, width)"
        #obs_patches = input[:,0]
        #next_obs_patches = input[:,1]
        
        #obs_embed = self.encoder(obs_patches).flatten(1)
        #next_obs_embed = self.encoder(next_obs_patches).flatten(1)
        
        #merged_embed = torch.cat([obs_embed, next_obs_embed], dim=1)
        
        #return self.mlp_head(merged_embed)                
        input_stack = einops.rearrange(input, 'b t c  h w -> b (t c) h w')
        input_enc = self.encoder(input_stack).flatten(1)
        return self.mlp_head(input_enc)    
        
