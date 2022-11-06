import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks.transformer import Transformer
from.networks.cnns import MLP
import einops
import math 


class TokenActionEmbedding(nn.Module):
    def __init__(self, obs_vocab_size, num_actions,action_dim, embed_dim):
        super(TokenActionEmbedding, self).__init__()

        self.num_actions = num_actions
        self.action_dim = action_dim
        self.obs_embedding = nn.Embedding(obs_vocab_size, embed_dim)
        self.action_embedding = nn.Embedding(self.num_actions*self.action_dim, embed_dim)
        
        
    def forward(self, tokens,action=None):
        """
        :param patches_list: batch of list list of patches 
        :param actions: list of actions, already one_hot encoded for discrete actions
        :return: list of patches with actions embedded
        """
        
        
        #inputs[:,:patches.shape[1],:patches.shape[2]] = patches
            # output of one_hot is shape (batch_size, num_classes)
            #inputs[:,patches.shape[1]:,patches.shape[2]:] = action.to(action.device)#torch.diag_embed(action) 
        embed_tokens = self.obs_embedding(tokens)
        action_in_vocab = action.long() + torch.arange(0,self.num_actions).long().to(action.device)*self.action_dim
        embed_actions = self.action_embedding(action_in_vocab.long())

        transformer_input = torch.cat((embed_tokens,embed_actions),dim=1)
        return transformer_input
    
    
class DiscreteAttentionDynamicsModel(nn.Module):
    def __init__(self, obs_vocab_size, num_actions, embed_dim, num_patches=16, num_attention_layers=2, mlp_dim=512, num_heads=2, dropout=0.,
                        group_actions=False,residual=True,use_attn_mask=False,regularizer_weight=0.,action_dim=1,discrete_actions=False,in_features=0):
        super(DiscreteAttentionDynamicsModel, self).__init__()

        self.regularizer_weight = regularizer_weight
        self.group_actions = group_actions
        # One additional dimension per action
        self.use_attn_mask = use_attn_mask

        self.token_action_embedding = TokenActionEmbedding(obs_vocab_size, embed_dim=embed_dim,num_actions=num_actions,action_dim=action_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches +num_actions, embed_dim).normal_(std=0.02))
        self.transformer = Transformer(dim=embed_dim, depth=num_attention_layers, heads=num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_dim,dropout=dropout,residual=residual)       
        self.obs_head = MLP(embed_dim, in_features, layer_sizes=[embed_dim])
        
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
            self.attn_mask = torch.ones(num_patches + self.action_embedding.action_dim, num_patches + self.action_embedding.action_dim)
            self.attn_mask[:patch_masks.shape[0],:patch_masks.shape[1]] = patch_masks
            self.attn_mask = self.attn_mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        # x is a list of [obs tokens, actions]
        # obs tokens are of shape (batch_size, tokens)
        # actions are of shape (batch_size, num_actions)
        
        self.pos_embedding = self.pos_embedding.to(x[0].device)
        if self.use_attn_mask:
            self.attn_mask = self.attn_mask.to(x[0].device)
        tokens, action = x[0], x[1]
                            
        inputs = self.token_action_embedding(tokens,action)
        inputs += self.pos_embedding
        out, attention_weights = self.transformer(inputs,attn_mask=self.attn_mask)
        
        out = out[:,:tokens.shape[1]] 
        out = self.obs_head(out)
        
        return out, attention_weights
    
    def compute_loss(self,encoding,actions):
        
        # encodings are of shape (batch_size, tokens)
        
        forwards_pred = []
        attention_weights = []
        
        # NO LONGER RECURSIVE - JUST PREDICTS NEXT STATE
        for step in range(1,actions.shape[1]):
            forward_encodings, attn_weight = self([encoding[:,step-1,:], actions[:,step-1]])
            forwards_pred.append(forward_encodings)
            attention_weights.append(attn_weight)
        forwards_pred = torch.stack(forwards_pred, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        
        if self.regularizer_weight > 0:
            regularizer_loss = (attention_weights.diagonal(dim1=-2,dim2=-1)).pow(2).mean()
        else:
            regularizer_loss = actions.new_zeros(1)
        
        forwards_pred = einops.rearrange(forwards_pred, 'b t n c -> (b t n )c')
        labels = einops.rearrange(encoding[:,1:,:], 'b t n -> (b t n)')
        forward_loss = F.cross_entropy(forwards_pred, labels)
        return forward_loss + self.regularizer_weight*regularizer_loss
