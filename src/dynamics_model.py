import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks.transformer import Transformer
import math 

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class AttentionDynamicsModel(nn.Module):
    def __init__(self, in_features, action_dim, num_patches=16, num_attention_layers=2, mlp_dim=128, num_heads=2, dropout=0., group_actions=False,residual=True,use_attn_mask=False):
        super(AttentionDynamicsModel, self).__init__()

        self.action_dim = action_dim
        self.transformer_dim = in_features + action_dim
        self.group_actions = group_actions
        # One additional dimension per action
        self.use_attn_mask = use_attn_mask
        if self.group_actions:
            self.num_actions = action_dim//2
        else:
            self.num_actions = action_dim
            
        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + self.num_actions, self.transformer_dim))
        #self.pos_embedding = positionalencoding1d(self.transformer_dim, num_patches + self.num_actions)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + self.num_actions, self.transformer_dim).normal_(std=0.02))
        self.action_embed = nn.Linear(action_dim, self.transformer_dim)
        self.transformer = Transformer(dim=self.transformer_dim, depth=num_attention_layers, heads=num_heads, dim_head=self.transformer_dim//num_heads, mlp_dim=mlp_dim,dropout=dropout,residual=residual)       
        
        self.attn_mask = None
        if self.use_attn_mask:
            patch_masks = []
            num_patch_sqrt = int(math.sqrt(num_patches))
            for x in range(num_patch_sqrt):
                for y in range(num_patch_sqrt):
                    mask = torch.zeros(num_patch_sqrt,num_patch_sqrt)
                    for i in range(-1,2):
                        for j in range(-1,2):
                            mask[max(min(x+i,num_patch_sqrt-1),0),max(min(y+j,num_patch_sqrt-1),0)] = 1
                    patch_masks.append(mask.flatten())
            patch_masks = torch.stack(patch_masks)
            self.attn_mask = torch.ones(num_patches + self.num_actions, num_patches + self.num_actions)
            self.attn_mask[:patch_masks.shape[0],:patch_masks.shape[1]] = patch_masks
            self.attn_mask = self.attn_mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        self.pos_embedding = self.pos_embedding.to(x[0].device)
        if self.use_attn_mask:
            self.attn_mask = self.attn_mask.to(x[0].device)
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
        
        inputs = torch.zeros(patches.shape[0], patches.shape[1] + self.num_actions, self.transformer_dim).to(patches.device)

        inputs[:,:patches.shape[1],:patches.shape[2]] = patches
        #action_diag = torch.cat((torch.zeros(patches.shape[0],action.shape[1]),torch.diag_embed(action)))
        if self.group_actions:
            for i in range(self.num_actions):
                inputs[:,patches.shape[1]+i,patches.shape[2]+(i*2):patches.shape[2]+(i*2)+2] = action[:,2*i:2*i+2]
        else:
            inputs[:,patches.shape[1]:,patches.shape[2]:] = torch.diag_embed(action)
        inputs += self.pos_embedding
        out, attention_weights = self.transformer(inputs,attn_mask=self.attn_mask)
        
        out = out[:,:patches.shape[1],:patches.shape[2]] 
        
        return out, attention_weights
