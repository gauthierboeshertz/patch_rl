import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks.transformer import Transformer
import math 
from .networks.cnns import MLP

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


class ObsActionEmbedding(nn.Module):
    def __init__(self, num_actions,action_dim, patchdes_dim,emb_dim, group_actions=True,discrete_actions=False):
        super(ObsActionEmbedding, self).__init__()
        if group_actions and discrete_actions:
            raise ValueError("Cannot group discrete actions")
        self.discrete_actions = discrete_actions
        self.group_actions = group_actions
        self.emb_dim = emb_dim
        
        if not self.discrete_actions:
            if self.group_actions:
                self.action_dim = action_dim//2
            else:
                self.action_dim = action_dim
            self.num_actions = self.action_dim
        else:
            self.num_actions = num_actions
            self.action_dim = action_dim # 4 discrete actions
            
        act_vocab =  self.num_actions * self.action_dim
        self.obs_embedding = nn.Linear(patchdes_dim, self.emb_dim)
        self.action_embedding = nn.Embedding(act_vocab, self.emb_dim)
        
        
    def forward(self, patches,action=None):
        """
        :param patches_list: batch of list list of patches 
        :param actions: list of actions, already one_hot encoded for discrete actions
        :return: list of patches with actions embedded
        """
        
        
        inputs = torch.zeros(patches.shape[0], patches.shape[1] + self.num_actions, self.emb_dim).to(patches.device)
        
        emb_patches = self.obs_embedding(patches)
        inputs[:,:patches.shape[1]] = emb_patches
        
        
        if not self.discrete_actions:
            if self.group_actions:
                for i in range(self.num_actions):
                    inputs[:,patches.shape[1]+i,patches.shape[2]+(i*2):patches.shape[2]+(i*2)+2] = action[:,2*i:2*i+2]
            else:
                inputs[:,patches.shape[1]:,patches.shape[2]:] = torch.diag_embed(action)
        else:
            # output of one_hot is shape (batch_size, num_classes)
            #inputs[:,patches.shape[1]:,patches.shape[2]:] = action.to(action.device)#torch.diag_embed(action) 
            action_in_vocab = action.long() + torch.arange(0,self.num_actions).long().to(action.device)*self.action_dim
            action_embed = self.action_embedding(action_in_vocab.long())
            inputs[:,patches.shape[1]:] = action_embed
        return inputs
    
    
class AttentionDynamicsModel(nn.Module):
    def __init__(self, in_features, num_actions,action_dim=4,embed_dim=256, num_patches=16, num_attention_layers=2, mlp_dim=128, num_heads=2, dropout=0.,
                        group_actions=False,residual=True,discrete_actions=False,use_attn_mask=False,regularizer_weight=0.):
        super(AttentionDynamicsModel, self).__init__()

        assert discrete_actions, "Only discrete actions are supported for now"
        
        self.regularizer_weight = regularizer_weight
        self.action_dim = action_dim
        self.group_actions = group_actions
        # One additional dimension per action
        self.use_attn_mask = use_attn_mask
        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + self.num_actions, self.transformer_dim))
        #self.pos_embedding = positionalencoding1d(self.transformer_dim, num_patches + self.num_actions)
        self.action_embedding = ObsActionEmbedding(num_actions,action_dim, in_features,embed_dim, group_actions=group_actions,discrete_actions=discrete_actions)
        
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + self.action_embedding.num_actions, embed_dim).normal_(std=0.02))
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
        self.pos_embedding = self.pos_embedding.to(x[0].device)
        if self.use_attn_mask:
            self.attn_mask = self.attn_mask.to(x[0].device)
        patches, action = x[0], x[1]
                            
        inputs = self.action_embedding(patches,action)
        inputs += self.pos_embedding
        out, attention_weights = self.transformer(inputs,attn_mask=self.attn_mask)
        
        out = out[:,:patches.shape[1]] 
        
        out = self.obs_head(out)
        
        return out, attention_weights
    
    def compute_loss(self,encoding,actions):
        #transformed_next_obs = self.transform(next_obs, self.transforms, augment=False)

        #use the first patch and use dynamics to predict the next patch recursively
        dynamic_input = encoding[:,0,:]
        forwards_pred = []
        attention_weights = []
        for step in range(1,actions.shape[1]):
            forward_encodings, attn_weight = self([dynamic_input, actions[:,step-1]])
            dynamic_input = forward_encodings
            forwards_pred.append(forward_encodings)
            attention_weights.append(attn_weight)
        forwards_pred = torch.stack(forwards_pred, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        
        if self.regularizer_weight > 0:
            regularizer_loss = (attention_weights.diagonal(dim1=-2,dim2=-1)).pow(2).mean()
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
        #decoded_patches, next_image_patches = self.remove_empty_patches_for_loss(forwards_pred, next_image_patches) 
        #
        return torch.abs(forwards_pred - encoding[:,1:]).mean() + self.regularizer_weight*regularizer_loss
