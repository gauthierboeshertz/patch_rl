import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks.transformer import Transformer
import math 
from .networks.cnns import MLP
import einops
from .mask_utils import make_gt_causal_mask, attn_rollout
import numpy as np

def huber_loss(pred, target, delta=1e-3, reduction="mean"):
    """Computes the Huber loss."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    loss = torch.where(abs_diff < delta,
                       0.5 * diff**2,
                       delta * (abs_diff - 0.5 * delta))
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss



class ActionDiscretizer():
    def __init__(self,num_discrete_bins) -> None:
        self.num_discrete_bins = num_discrete_bins
        self.bins = torch.linspace(-1,1,self.num_discrete_bins)
        self.action_dim =  self.num_discrete_bins ** 2
        
    def discretize(self,actions):
        assert actions.shape[2] == 2 , "actions should be of shape (batch_size,num_actions,2)"
        disc_actions = actions.new_zeros((actions.shape[0],actions.shape[1]))
        self.bins = self.bins.to(actions.device)
        bucket_actions_x = torch.bucketize(actions[:, :, 0], self.bins)
        bucket_actions_y = torch.bucketize(actions[:, :, 1], self.bins)
        disc_actions = bucket_actions_x * self.num_discrete_bins + bucket_actions_y
        return disc_actions
       #torch.stack([torch.bucketize(actions[:,i]) for i in range(self.num_actions)],dim=1)



class ObsActionEmbedding(nn.Module):
    def __init__(self, num_actions,action_dim, patchdes_dim,emb_dim, group_actions=True,
                    discrete_actions=False,discretize_actions=False,num_discrete_bins=16) -> None:
        super(ObsActionEmbedding, self).__init__()
        if group_actions and discrete_actions:
            raise ValueError("Cannot group discrete actions")
        self.discrete_actions = discrete_actions
        self.group_actions = group_actions
        self.emb_dim = emb_dim
        self.discretize_actions = discretize_actions
        if self.discretize_actions:
            self.action_discretizer = ActionDiscretizer(num_discrete_bins)
        self.action_dim = action_dim
        
        if self.group_actions:
            self.num_action_per_group = 2
        else:
            self.num_action_per_group = 1

        if self.discretize_actions:
            self.num_actions = self.action_dim // self.num_action_per_group
            act_vocab = self.action_discretizer.action_dim * self.num_actions
            self.action_embedding = nn.Embedding(act_vocab, self.emb_dim)
        elif not self.discrete_actions:
            assert self.action_dim % self.num_action_per_group == 0, "action_dim must be divisible by 2 if group_actions is True"
            self.num_actions = self.action_dim // self.num_action_per_group
            self.action_embeddings = nn.ModuleList([nn.Linear(self.num_action_per_group, self.emb_dim,bias=False) for _ in range(self.num_actions)])
        else:
            self.num_actions = num_actions
            act_vocab =  self.num_actions * self.action_dim
            self.action_embedding = nn.Embedding(act_vocab, self.emb_dim)
        
        self.obs_embedding = nn.Linear(patchdes_dim, self.emb_dim,bias=False)
        
    def forward(self, patches,action=None):
        """
        :param patches_list: batch of list list of patches 
        :param actions: list of actions, already one_hot encoded for discrete actions
        :return: list of patches with actions embedded
        """
        
        
        inputs = torch.zeros(patches.shape[0], patches.shape[1] + self.num_actions, self.emb_dim).to(patches.device)
        emb_patches = self.obs_embedding(patches)
        inputs[:,:patches.shape[1]] = emb_patches
        
        if self.discretize_actions:
            discretized_actions = self.action_discretizer.discretize(einops.rearrange(action,'b (n a) -> b n a',n=self.num_actions))
            action_in_vocab = discretized_actions.long() + torch.arange(0,self.num_actions).long().to(action.device)*self.action_discretizer.action_dim
            embedded_actions = self.action_embedding(action_in_vocab.long())

        elif not self.discrete_actions:
            embedded_actions = []
            for action_idx in range(self.num_actions):
                action_to_embed = action[:,action_idx*self.num_action_per_group:(action_idx+1)*self.num_action_per_group]
                embedded_actions.append(self.action_embeddings[action_idx](action_to_embed))
            embedded_actions = torch.stack(embedded_actions,dim=1)
            
        else:
            # output of one_hot is shape (batch_size, num_classes)
            #inputs[:,patches.shape[1]:,patches.shape[2]:] = action.to(action.device)#torch.diag_embed(action) 
            action_in_vocab = action.long() + torch.arange(0,self.num_actions).long().to(action.device)*self.action_dim
            embedded_actions = self.action_embedding(action_in_vocab.long())
        
        inputs[:,patches.shape[1]:] = embedded_actions
        return inputs
    
    
class AttentionDynamicsModel(nn.Module):
    def __init__(self, in_features, num_actions,action_dim=4,embed_dim=256, num_patches=16, num_attention_layers=2, mlp_dim=128, num_heads=2, dropout=0.,
                        group_actions=False,residual=True,discrete_actions=False,use_attn_mask=False,regularizer_weight=0.,device="cpu",
                        use_gt_mask=False,action_regularization_weight=0.1,temperature=1.,end_residual=False,head_disagreement_weight=0,causal_mask_threshold=0.2,
                        head_fusion="max",discard_ratio=0.98,discretize_actions=False,num_discrete_bins=16, predict_rewards=False,num_rewards=5, reward_loss_weight=0) -> None:
        super(AttentionDynamicsModel, self).__init__()


        self.regularizer_weight = regularizer_weight
        self.action_dim = action_dim
        self.group_actions = group_actions
        self.use_gt_mask = use_gt_mask
        self.action_regularization_weight = action_regularization_weight
        self.head_disagreement_weight = head_disagreement_weight
        self.end_residual = end_residual
        self.causal_mask_threshold = causal_mask_threshold
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        # One additional dimension per action
        self.use_attn_mask = use_attn_mask
        self.residual = residual
        self.device = device
        self.action_embedding = ObsActionEmbedding(num_actions,action_dim, in_features,embed_dim, group_actions=group_actions,discrete_actions=discrete_actions,discretize_actions=discretize_actions,num_discrete_bins=num_discrete_bins)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + self.action_embedding.num_actions + (1 if predict_rewards else 0), embed_dim).normal_(std=0.02))
        self.transformer = Transformer(dim=embed_dim, depth=num_attention_layers, heads=num_heads, dim_head=embed_dim//num_heads,
                                       mlp_dim=mlp_dim,dropout=dropout,residual=residual,temperature=temperature)       
        self.obs_head = MLP(embed_dim, in_features, layer_sizes=[embed_dim])
        self.predict_rewards = predict_rewards
        self.reward_loss_weight = reward_loss_weight
        self.reward_head = MLP(embed_dim, num_rewards, layer_sizes=[embed_dim])
        self.reward_token = nn.Parameter(torch.empty(1, 1, embed_dim).normal_(std=0.02))    
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
            self.attn_mask = torch.ones(num_patches + self.action_embedding.num_actions, num_patches + self.action_embedding.num_actions)
            self.attn_mask[:patch_masks.shape[0],:patch_masks.shape[1]] = patch_masks
            self.attn_mask[patch_masks.shape[0]:, :-self.action_embedding.num_actions] = 0
            self.attn_mask = self.attn_mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        #self.pos_embedding = self.pos_embedding.to(x[0].device)
        if self.use_attn_mask:
            self.attn_mask = self.attn_mask.to(x[0].device)
        patches, action = x[0], x[1]
                                            
        inputs = self.action_embedding(patches,action)
        if self.predict_rewards:
            inputs = torch.cat([inputs,self.reward_token.repeat(inputs.shape[0],1,1)],dim=1)
        inputs += self.pos_embedding
        out, attention_weights = self.transformer(inputs,attn_mask=None)
        
        if self.predict_rewards:
            reward = self.reward_head(out[:,-1,:])
        else:
            reward = None
        out = out[:,:patches.shape[1]]  # remove action embedding
        out = self.obs_head(out)
        attention_weights = attention_weights[:,:,:, :inputs.shape[1] -( 1 if self.predict_rewards else 0), :inputs.shape[1]-( 1 if self.predict_rewards else 0)]
        return out, reward, attention_weights
            
    
    def get_attn_weights(self, images, actions,encoder=None):
        """
        get_causal_mask returns a causal mask from transitions.

        _extended_summary_

        Args:
            images (_type_): Preprocessed batch of pre and post images, shape (batch_size, 3, H,W)
            actions (_type_): Batch of actions, shape (batch_size, num_actions, action_dim)
        """
        if encoder is None:
            encodings = images
        else:
            encodings = encoder.get_encoding_for_dynamics(images)
            encodings = einops.rearrange(encodings, "(b n) c -> b n c", b=images.shape[0])
            
        attention_weights = self.forward([encodings,actions])[-1]
        return attention_weights
    
            
    def get_causal_mask(self, images, actions,encoder=None):
        """
        get_causal_mask returns a causal mask from transitions.

        _extended_summary_

        Args:
            images (_type_): Preprocessed batch of pre and post images, shape (batch_size,  3, H,W)
            actions (_type_): Batch of actions, shape (batch_size, num_actions, action_dim)
        """
        attn_weights = self.get_attn_weights(images,actions,encoder=encoder)
        
        causal_masks = []
        
        for b in range(attn_weights.shape[0]):
            causal_masks.append(attn_rollout(attn_weights[b], discard_ratio=self.discard_ratio,head_fusion=self.head_fusion,residual=self.residual))
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
        rewards = []
        for step in range(1,actions.shape[1]):
            forward_encodings, reward, attn_weight = self([dynamic_input, actions[:,step-1]])
            dynamic_input = forward_encodings
            forwards_pred.append(forward_encodings)
            rewards.append(reward)
            attention_weights.append(attn_weight)
        forwards_pred = torch.stack(forwards_pred, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        
        regularizer_loss = actions.new_zeros(1)
        
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

            regularizer_loss = self.regularizer_weight*regularizer_loss

        # attention_weights shape  B,T,LAYERS, NUM HEADS,ROW,COL([32, 2, 1, 8, 68, 68])
        action_reg_loss = actions.new_zeros(1)
        if self.action_regularization_weight > 0:
            #action_reg_loss = attention_weights[..., -self.action_embedding.num_actions:].mean()
            action_reg_loss = ( (attention_weights[..., -self.action_embedding.num_actions:] >0.1).sum(dim=-2) >2 ).float().mean()
            action_reg_loss = self.action_regularization_weight*action_reg_loss
            
        head_disagreement_loss = actions.new_zeros(1)
        
        if self.head_disagreement_weight > 0:
            head_disagreement_loss =  self.head_disagreement_weight*attention_weights.prod(dim=-3).mean()
            
        loss_dict = {}
        
        #if encoder_decoder is None:
        forward_loss = F.mse_loss(forwards_pred,encoding[:,1:])#torch.abs(forwards_pred - encoding[:,1:]).mean()
        loss_dict["forward_loss"] = float(forward_loss)
        
        if self.predict_rewards:
            rewards = torch.stack(rewards, dim=1)
            rewards = einops.rearrange(rewards, "b t n -> (b t) n")
            lab_rewards = einops.rearrange(batch[2][:,:-1].to(self.device).long(), "b t -> (b t)")
            rewards_loss = F.cross_entropy(rewards,lab_rewards)*self.reward_loss_weight
        else:
            rewards_loss = 0
            
        if encoder_decoder is not None:
            forwards_pred = einops.rearrange(forwards_pred, "b t n c -> (b t n) c")
            pred_next_patches = encoder_decoder.decode(forwards_pred)
            #pred_next_images = einops.rearrange(pred_next_images, "(b t n) c h w -> (b t n) c h w", b=encoding.shape[0], t=encoding.shape[1]-1)
            label_next_images = (batch[2][:,1:].to(self.device).float())/255.
            label_next_patches = einops.rearrange(label_next_images, "b t c (h p1) (w p2) -> (b t h w) c p1 p2", p1=encoder_decoder.patch_size, p2=encoder_decoder.patch_size)
            
            recons_loss = F.mse_loss(pred_next_patches,label_next_patches)#torch.abs(pred_next_images - label_next_patches).mean()
            loss_dict["forward_recons_loss"] = float(recons_loss)
            forward_loss += recons_loss
        
        loss_dict["rewards_loss"] = float(rewards_loss)
        loss_dict["regularizer_loss"] = float(regularizer_loss)
        loss_dict["head_disagreement_loss"] = float(head_disagreement_loss)
        loss_dict["action_reg_loss"] = float(action_reg_loss)
        
        return forward_loss + regularizer_loss+ action_reg_loss + head_disagreement_loss + rewards_loss, loss_dict
