import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks.transformer import Transformer
import math 
from .networks.cnns import MLP
import einops
from .mask_utils import make_gt_causal_mask, attn_rollout
import numpy as np

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
                        group_actions=False,residual=True,discrete_actions=False,use_attn_mask=False,regularizer_weight=0.,same_action_prediction=False,device="cpu",
                        use_gt_mask=False,action_regularization=False):
        super(AttentionDynamicsModel, self).__init__()

        assert discrete_actions, "Only discrete actions are supported for now"
        
        self.regularizer_weight = regularizer_weight
        self.action_dim = action_dim
        self.group_actions = group_actions
        self.same_action_prediction = same_action_prediction
        self.use_gt_mask = use_gt_mask
        self.action_regularization = action_regularization
        # One additional dimension per action
        self.use_attn_mask = use_attn_mask
        self.residual = residual
        self.device = device
        self.action_embedding = ObsActionEmbedding(num_actions,action_dim, in_features,embed_dim, group_actions=group_actions,discrete_actions=discrete_actions)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + self.action_embedding.num_actions, embed_dim).normal_(std=0.02))
        self.transformer = Transformer(dim=embed_dim, depth=num_attention_layers, heads=num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_dim,dropout=dropout,residual=residual)       
        self.obs_head = MLP(embed_dim, in_features, layer_sizes=[embed_dim])
        
        self.action_head = MLP(embed_dim, action_dim*num_actions, layer_sizes=[embed_dim])
        
        self.attn_mask = None
        attn_mask_neighbor_size = 2
        #if self.use_attn_mask:
        if self.regularizer_weight > 0 or self.use_attn_mask:
            patch_masks = []
            num_patch_sqrt = int(math.sqrt(num_patches))
            for x in range(num_patch_sqrt):
                for y in range(num_patch_sqrt):
                    mask = torch.zeros(num_patch_sqrt,num_patch_sqrt)
                    for i in range(-attn_mask_neighbor_size+1,attn_mask_neighbor_size):
                        for j in range(-attn_mask_neighbor_size +1,2):
                            mask[max(min(x+i,num_patch_sqrt-1),0),max(min(y+j,num_patch_sqrt-1),0)] = 1
                    patch_masks.append(mask.flatten())
            patch_masks = torch.stack(patch_masks)
            self.attn_mask = torch.ones(num_patches + self.action_embedding.action_dim, num_patches + self.action_embedding.action_dim)
            self.attn_mask[:patch_masks.shape[0],:patch_masks.shape[1]] = patch_masks
            self.attn_mask[patch_masks.shape[0]:, :-self.action_embedding.action_dim] = 0
            self.attn_mask = self.attn_mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        #self.pos_embedding = self.pos_embedding.to(x[0].device)
        if self.use_attn_mask:
            self.attn_mask = self.attn_mask.to(x[0].device)
        patches, action = x[0], x[1]
                            
        inputs = self.action_embedding(patches,action)
        inputs += self.pos_embedding
        out, attention_weights = self.transformer(inputs,attn_mask=(self.attn_mask if self.use_attn_mask else None))#self.attn_mask)
        
        if not self.same_action_prediction:
            out = out[:,:patches.shape[1]]  # remove action embedding
            out = self.obs_head(out)
            return out, attention_weights
        else:
            out_obs = self.obs_head(out[:,:patches.shape[1]])  # remove action embedding
            out_act  = self.action_head(out[:,patches.shape[1]:])
            return out_obs, out_act, attention_weights
    
            
    
    def get_attn_weights(self, images, actions,encoder=None):
        """
        get_causal_mask returns a causal mask from transitions.

        _extended_summary_

        Args:
            images (_type_): Preprocessed batch of pre and post images, shape (batch_size, 2, 3, H,W)
            actions (_type_): Batch of actions, shape (batch_size, num_actions, action_dim)
        """
        if encoder is None:
            encodings = images
        else:
            encodings = encoder.get_encoding_for_dynamics(images)
            encodings = einops.rearrange(encodings, "(b n) c -> b n c", b=images.shape[0])
            
        attention_weights = self.forward([encodings,actions])[-1]
        return attention_weights
    
            
    def get_causal_mask(self, images, actions,encoder=None, discard_ratio=0.8,head_fusion='mean'):
        """
        get_causal_mask returns a causal mask from transitions.

        _extended_summary_

        Args:
            images (_type_): Preprocessed batch of pre and post images, shape (batch_size, 2, 3, H,W)
            actions (_type_): Batch of actions, shape (batch_size, num_actions, action_dim)
        """
        attn_weights = self.get_attn_weights(images,actions,encoder=encoder)
        
        causal_masks = []
        
        for b in range(attn_weights.shape[0]):
            causal_masks.append(attn_rollout(attn_weights[b], discard_ratio=discard_ratio,head_fusion=head_fusion,residual=self.residual))
        causal_masks = torch.stack(causal_masks)
        causal_masks[:,-self.action_embedding.num_actions:,:] = 0
        return causal_masks, attn_weights
        
        
    def compute_loss(self,batch,encoder_decoder=None):
        #use the first patch and use dynamics to predict the next patch recursively
        encoding = batch[0].to(self.device)
        actions = batch[1].to(self.device)
        
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
            """
            if self.use_gt_mask:
                gt_masks = batch[-1].to(self.device)
                regularizer_loss = 0
                for t in range(gt_masks.shape[1]):
                    pred_masks = self._get_causal_mask_from_attn_weights(attention_weights[:,t])
                    regularizer_loss +=  (1 - pred_masks[gt_masks[:,t]== 1]).mean()
            else:
            """
            self.attn_mask = self.attn_mask.to(encoding.device)
            regularizer_loss = ((1- self.attn_mask)*attention_weights).mean()
            
        else:
            regularizer_loss = actions.new_zeros(1)
        
        loss_dict = {}
        
        
        if encoder_decoder is None:
            forward_loss = torch.abs(forwards_pred - encoding[:,1:]).mean()
            loss_dict["forward_loss"] = float(forward_loss)
        else:
            forwards_pred = einops.rearrange(forwards_pred, "b t n c -> (b t n) c")
            pred_next_images = encoder_decoder.decode(forwards_pred)
            pred_next_images = einops.rearrange(pred_next_images, "(b t n) c h w -> (b t n) c h w", b=encoding.shape[0], t=encoding.shape[1]-1)
            label_next_images = (batch[2][:,1:].to(self.device))/255.0
            label_next_patches = einops.rearrange(label_next_images, "b t c (h p1) (w p2) -> (b t h w) c p1 p2", p1=encoder_decoder.patch_size, p2=encoder_decoder.patch_size)
            forward_loss = torch.abs(pred_next_images - label_next_patches).mean()
            loss_dict["forward_recons_loss"] = float(forward_loss)
        
        
        loss_dict["regularizer_loss"] = float(regularizer_loss)
        
        return forward_loss + self.regularizer_weight*regularizer_loss, loss_dict
