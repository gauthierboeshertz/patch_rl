import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks.transformer import Transformer


class AttentionDynamicsModel(nn.Module):
    def __init__(self, in_features, action_dim, num_patches=16, num_attention_layers=2, mlp_dim=128, num_heads=2, dropout=0., object_disentanglement=None,):
        super(AttentionDynamicsModel, self).__init__()

        self.action_dim = action_dim
        self.transformer_dim = in_features + action_dim
        
        # One additional dimension per action
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + action_dim, self.transformer_dim))

        self.transformer = Transformer(dim=self.transformer_dim, depth=num_attention_layers, heads=num_heads, dim_head=self.transformer_dim//num_heads, mlp_dim=mlp_dim,dropout=dropout)        
        
    def forward(self, x):
        
        patches, action = x[0], x[1]
        
        """
        if self.object_disentanglement:
            obj_x = torch.zeros(x.shape[0],len(self.object_disentanglement),self.action_dim).to(x.device)
            for obj_idx, obj in enumerate(self.object_disentanglement):
                obj_x[:,obj_idx,obj] = x[:,obj]
            x = obj_x
            
            then 
            
        if self.object_disentanglement:
            x = x[:,:len(self.object_disentanglement)//2,]
            entangled_x = torch.zeros(x.shape[0], 1).to(x.device)
            for obj_idx, obj in enumerate(self.object_disentanglement[:len(self.object_disentanglement)//2]):
                entangled_x[:,obj] = x[:,obj_idx,obj]
            x = entangled_x

        """
        inputs = torch.zeros(patches.shape[0], patches.shape[1] + self.action_dim, self.transformer_dim).to(patches.device)

        inputs[:,:patches.shape[1],:patches.shape[2]] = patches
        #action_diag = torch.cat((torch.zeros(patches.shape[0],action.shape[1]),torch.diag_embed(action)))
        inputs[:,patches.shape[1]:,patches.shape[2]:] = torch.diag_embed(action)
        inputs += self.pos_embedding
        out, attention_weights = self.transformer(inputs)
        
        out = out[:,:patches.shape[1],:patches.shape[2]] 
        
        return out, attention_weights
