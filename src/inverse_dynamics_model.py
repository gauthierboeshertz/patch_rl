import torch
from torch import nn
import torch.nn.functional as F
import einops
import math
from .networks.cnns import Conv2dModel, MLP

class PatchInverseDynamicsModel(nn.Module):
    """
    Predict action from the observation and the next observation.
    """

    def __init__(
            self,
            encoder,
            mlp,
            discrete_action_space,
            num_actions,
            ):
        super(PatchInverseDynamicsModel,self).__init__()
        
        print(encoder)
        self.encoder = Conv2dModel(**encoder)
        self.mlp_head = MLP(**mlp)
        self.discrete_action_space = discrete_action_space
        self.num_actions = num_actions
        
    def forward(self, input):
        """ Predicts the action from the observation and the next observation with patch embeddings
        """
        assert len(input.shape) == 5, "The input should be of shape (batch_size,2, patch_dim, height, width)"
        assert input.shape[1] == 2, f"Time dimension should be 2, but whole shape is {input.shape}"

         
        input_stack = einops.rearrange(input, 'b t c  h w -> b (t c) h w')
        input_enc = self.encoder(input_stack).flatten(1)
        return self.mlp_head(input_enc)    
        
    def compute_loss(self, encodings, actions):
        num_patches_sqrt = int(math.sqrt(encodings.shape[-2]))
        
        encodings = einops.rearrange(encodings, "b t (h w) c -> b t c h w", h=num_patches_sqrt, w=num_patches_sqrt)
        obs_pairs = torch.stack([encodings[:,:-1], encodings[:,1:]], dim=2)
        obs_pairs = einops.rearrange(obs_pairs, "b t p c h w -> (b t) p c h w", h=num_patches_sqrt, w=num_patches_sqrt)
        pred_actions = self(obs_pairs)
        actions = actions[:,:-1]
        actions = einops.rearrange(actions, "b t a -> (b t) a ")
        if self.discrete_action_space:
            pred_actions = einops.rearrange(pred_actions, "b (a h) -> b a h", a=self.num_actions)
            return F.cross_entropy(pred_actions, actions.long())
        else:
            return F.mse_loss(pred_actions, actions)
