from moog_demos.example_configs import bouncing_sprites
import torch
EPS = 1e-6
import numpy as np
from scipy.sparse.csgraph import connected_components


TARGET_POS_IMG = [[9,16,22,29],
                  [35,42,22,29],
                  [86,93,48,55],
                  [86,93,73,80]]

TARGET_MASK = torch.zeros((4,128,128))
TARGET_MASK[0,TARGET_POS_IMG[0][0]:TARGET_POS_IMG[0][1],TARGET_POS_IMG[0][2]:TARGET_POS_IMG[0][3]] = 1
TARGET_MASK[1,TARGET_POS_IMG[1][0]:TARGET_POS_IMG[1][1],TARGET_POS_IMG[1][2]:TARGET_POS_IMG[1][3]] = 1
TARGET_MASK[2,TARGET_POS_IMG[2][0]:TARGET_POS_IMG[2][1],TARGET_POS_IMG[2][2]:TARGET_POS_IMG[2][3]] = 1
TARGET_MASK[3,TARGET_POS_IMG[3][0]:TARGET_POS_IMG[3][1],TARGET_POS_IMG[3][2]:TARGET_POS_IMG[3][3]] = 1

from .patch_utils import image_to_patches, patches_to_image


def get_cc_from_mask(mask):
    """
    Converts a mask into a list of CC indices tuples.
    E.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
    this will return [array([0]), array([1]), array([2, 3])]

    Note that the mask should be a square, so in case we have (s, a) x (s2,),
    we should first dummy a2 columns to form a square mask. 
    """
    ccs = connected_components(mask)
    num_ccs, cc_idxs = ccs
    return [np.where(cc_idxs == i)[0] for i in range(num_ccs)]


def get_cc_dicts_from_mask(mask,num_patches,num_actions):
    """
    Converts a mask into a list of CC indices tuples.
    E.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
    this will return [array([0]), array([1]), array([2, 3])]

    Note that the mask should be a square, so in case we have (s, a) x (s2,),
    we should first dummy a2 columns to form a square mask. 
    """
    ccs = get_cc_from_mask(mask)
    patch_idx = list(range(num_patches))
    action_idx = list(range(num_patches,num_patches+num_actions))
    
    cc_dicts = []
    for cc in ccs:
        cc_dict = {}
        cc_dict["patches"] = [c for c in cc if c in patch_idx]
        cc_dict["actions"] = [c for c in cc if c in action_idx]
        cc_dicts.append(cc_dict)
    return cc_dicts
    
def indep_comp_in_other(c,ccs):
    return  not any([ len(cc["actions"]) >0  for cc in ccs if c in cc["patches"]])


def maybe_remove_component_with_action(action_c,ccs):
    return [cc for cc in ccs if (not ((action_c  in cc["actions"] ) and (len(cc["actions"]) == 1)))]


def swap_transition_components(patches1,patches2, action1,action2, next_patches1, next_patches2, cc1_to_swap,cc2_to_swap,group_actions=False):
    
    swapped_patches = []
    swapped_actions = []
    swapped_next_patches = []
    for cc1,cc2 in zip(cc1_to_swap,cc2_to_swap):
        
        # swap actions
        new_action1 = action1.clone()
        new_action2 = action2.clone()
        action1_idx = [act - patches1.shape[0] for act in cc1["actions"]]
        action2_idx = [act - patches1.shape[0] for act in cc2["actions"]]
        
        if group_actions:
            action1_idx_ = []
            for act_idx in action1_idx:
                action1_idx_.append(2*act_idx)
                action1_idx_.append(2*act_idx+1)
            action1_idx = action1_idx_
            action2_idx_ = []
            for act_idx in action2_idx:
                action2_idx_.append(2*act_idx)
                action2_idx_.append(2*act_idx+1)
            action2_idx = action2_idx_

        new_action1[action1_idx] = action2[action2_idx]
        new_action2[action2_idx] = action1[action1_idx]
        
        swapped_actions.append(new_action1)
        swapped_actions.append(new_action2)
        
        cc1_wo_act = cc1["patches"]
        cc2_wo_act = cc2["patches"]
        
        # swap current image patches
        new_patches1 = patches1.clone()
        new_patches2 = patches2.clone()
        
        new_patches1[cc1_wo_act], new_patches1[cc2_wo_act] = patches2[cc1_wo_act], patches2[cc2_wo_act]
        swapped_patches.append(new_patches1)
        new_patches2[cc1_wo_act], new_patches2[cc2_wo_act] = patches1[cc1_wo_act], patches1[cc2_wo_act]
        swapped_patches.append(new_patches2)

        # swap next image patches
        new_next_patches1 = next_patches1.clone()
        new_next_patches2 = next_patches2.clone()
        new_next_patches1[cc1_wo_act], new_next_patches1[cc2_wo_act] = next_patches2[cc1_wo_act], next_patches2[cc2_wo_act]
        swapped_next_patches.append(new_next_patches1)
        new_next_patches2[cc1_wo_act], new_next_patches2[cc2_wo_act] = next_patches2[cc1_wo_act], next_patches2[cc2_wo_act]
        swapped_next_patches.append(new_patches2)
        
    if swapped_patches:
        return swapped_patches, swapped_actions, swapped_next_patches
    else:
        return None
    

def get_ccs_to_swap(cc1, cc2):
    #cc1 = get_cc_dicts_from_mask(m1,num_patches=m1.shape[0]-num_actions,num_actions=num_actions)    
    #cc2 = get_cc_dicts_from_mask(m2,num_patches=m1.shape[0]-num_actions,num_actions=num_actions)   
    
    
    action_ccs1 = [cc for cc in cc1 if len(cc["actions"]) > 0]
    action_ccs2 = [cc for cc in cc2 if len(cc["actions"]) > 0]
    
    ccs1_to_swap = []
    ccs2_to_swap = []
    for act_cc1 in action_ccs1:
        
        can_swap_cc = True
        
        #assert len(act_cc1["patches"]) > 0, "action component without patches"
        
        if len(act_cc1["actions"]) != 1:
            continue
            
        act_cc1_action = act_cc1["actions"][0]
        cc2_same_act = [c for c in cc2 if act_cc1_action in c["actions"]][0]
        
        if len(cc2_same_act["actions"]) != 1:
            continue
       # assert len(cc2_same_act["patches"]) > 0, "action component without patches"

        cc1_wo_action_cc = maybe_remove_component_with_action(act_cc1_action,cc1)
        for c in cc2_same_act["patches"]:
            if not indep_comp_in_other(c,cc1_wo_action_cc):      
                can_swap_cc = False

        cc2_wo_action = maybe_remove_component_with_action(act_cc1_action,cc2)
        for c in act_cc1["patches"]:
            if not indep_comp_in_other(c,cc2_wo_action):      
                can_swap_cc = False 
                
        if not can_swap_cc:
            continue
        ccs1_to_swap.append(act_cc1)
        ccs2_to_swap.append(cc2_same_act)

    return ccs1_to_swap, ccs2_to_swap

    

def get_transition_data(dataset, t_idx,use_gt_mask,encoder=None,dynamics=None,patch_size=16,num_actions=4):
    image, action, next_image = dataset[t_idx][0], dataset[t_idx][1],dataset[t_idx][2]
    image = image[0]
    next_image = next_image[0]
    action = action[0]
    if use_gt_mask:
        images = torch.stack([image,next_image])
        mask = make_gt_causal_mask(images.unsqueeze(0),action.unsqueeze(0),
                                   patch_size=patch_size,num_sprites=num_actions)
    else:
        mask, _ = dynamics.get_causal_mask(image.unsqueeze(0)/255., action.unsqueeze(0),encoder=encoder,head_fusion="mean",discard_ratio=0.98)
        mask = mask > 0.1
    mask = mask.cpu().numpy().transpose(0,2,1)[0]
    
    cc = get_cc_from_mask(mask)
    return image, action, next_image, mask, cc    
    
    
def do_coda_on_transitions(transition1,transition2, patch_size = (16,16), num_actions=4,group_actions=True):
    """
    do_coda_on_transitions 

    _extended_summary_

    Args:
        transition1 (_type_): tupe of (image, action, next_image, reward,done, cc)
        transition2 (_type_): tupe of (image, action, next_image, reward, done, cc)
        patch_size (_type_): tuple of (patch_size, patch_size)
        num_actions (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    image1, action1, next_image1, _, _, cc1 = transition1 #get_transition_data(dataset, t1_idx,use_gt_mask=use_gt_mask,encoder=encoder,dynamics=dynamics,patch_size=patch_size,num_actions=num_actions)    
    image2, action2, next_image2, _, _, cc2 = transition2 #get_transition_data(dataset, t2_idx,use_gt_mask=use_gt_mask,encoder=encoder,dynamics=dynamics,patch_size=patch_size,num_actions=num_actions)
    
    patches1 = image_to_patches(image1, patch_size=patch_size,num_patches_sqrt=image1.shape[2]//patch_size[0])
    patches2 = image_to_patches(image2, patch_size=patch_size,num_patches_sqrt=image2.shape[2]//patch_size[0])

    next_patches1 = image_to_patches(next_image1, patch_size=patch_size,num_patches_sqrt=image1.shape[2]//patch_size[0])
    next_patches2 = image_to_patches(next_image2, patch_size=patch_size,num_patches_sqrt=image2.shape[2]//patch_size[0])

    comp_to_swap_ccs1, comp_to_swap_ccs2 = get_ccs_to_swap(cc1, cc2)
    
    coda_transitions = swap_transition_components(patches1,patches2, action1,action2, next_patches1, next_patches2, 
                                        comp_to_swap_ccs1,comp_to_swap_ccs2,group_actions=group_actions)
    
    if coda_transitions is not None:
        swapped_patches = patches_to_image(torch.stack(coda_transitions[0]), patch_size[0], image1.shape[-2])
        swapped_next_patches = patches_to_image(torch.stack(coda_transitions[2]), patch_size[0], image1.shape[-2])
        
        return swapped_patches, coda_transitions[1], swapped_next_patches
    else:
        return None



def gt_reward_function(image, next_image, num_sprites=4,touch_reward=0.25,no_touch_reward=-0.01):
    sprite_colors = torch.Tensor(bouncing_sprites.color_list).to(image.device)
    
    total_rewards = torch.zeros((image.shape[0])).to(image.device)
    perm_image = image.permute(0,2,3,1)
    perm_next_image = next_image.permute(0,2,3,1)
    for sprite_idx in range(num_sprites):
        sprite_color = sprite_colors[sprite_idx]
        sprite_in_image = ((perm_image == sprite_color).all(dim=-1)).sum( (1,2)) > 0
        sprite_in_next_image = ((perm_next_image == sprite_color).all(dim=-1)).sum((1,2)) > 0
        
        total_rewards += touch_reward*(~sprite_in_next_image & (sprite_in_image)).float()
    
    total_rewards[total_rewards == 0] = no_touch_reward
    
    return total_rewards

def gt_count_function(image, num_sprites=4):
    
    sprite_colors = torch.Tensor(bouncing_sprites.color_list).to(image.device)
    sprites_count = torch.zeros((image.shape[0])).to(image.device)
    perm_image = image.permute(0,2,3,1)
    for sprite_idx in range(num_sprites):
        sprite_color = sprite_colors[sprite_idx]
        sprite_in_image = ((perm_image == sprite_color).all(dim=-1)).sum( (1,2)) > 0

        sprites_count += sprite_in_image.float()
    
    return sprites_count


def remove_targets_from_image(image):
    new_image = image.clone()
    for target in TARGET_POS_IMG:
        new_image[:,:,target[0]:target[1],target[2]:target[3]] = 0
    return new_image

def image_to_image_patch_batch(image,patch_size,num_patches_sqrt=None):
    image_patches = []
    for i in range(num_patches_sqrt):
        for j in range(num_patches_sqrt):
            image_patches.append(image[:,:,i*patch_size[0]:(i+1)*patch_size[0],j*patch_size[1]:(j+1)*patch_size[1]])
    return torch.stack(image_patches,dim=1)


def get_sprites_in_patch(image,patch_size=16,num_sprites=4):
    no_targets_image = image.clone() #remove_targets_from_image(image) REPUT THIS IF SHOWING TARGETS
    sprite_colors = torch.Tensor(bouncing_sprites.color_list).to(no_targets_image.device)
    sprites_in_patch = torch.zeros(( image.shape[0],num_sprites,(no_targets_image.shape[2]//patch_size) * (image.shape[3]//patch_size)))
    for sprite_idx in range(num_sprites):
        sprite_color = sprite_colors[sprite_idx]
        sprite_image = ((no_targets_image.permute(0,2,3,1) == torch.Tensor(sprite_color)).permute(0,3,1,2)).all(dim=1,keepdim=True).float()
        sprite_image_patches = image_to_image_patch_batch(sprite_image,patch_size=(patch_size,patch_size),num_patches_sqrt=(image.shape[2]//patch_size))
        sprite_image_patches = sprite_image_patches.sum((-3,-2,-1)) > 0
        sprite_image_patches = sprite_image_patches
        sprites_in_patch[:,sprite_idx] = sprite_image_patches
    sprites_in_patch = sprites_in_patch.flatten(-1)
    return sprites_in_patch    


def make_gt_causal_mask(image,actions, next_image,patch_size=16,num_sprites=4):
    """
    make_causal_mask Takes two image, pre and post transition and returns a causal mask


    Args:
        images (_type_): tensor of shape (2,3,128,128)
        actions (_type_): tensor of shape (num_actions==num_sprites)
        patch_size (int, optional): Patch size in pixels. Defaults to 16.
        num_sprites (int, optional): number of sprites Defaults to 4.

    Returns:
        _type_: _description_
    """
    num_patches = (image.shape[-1]//patch_size) * (image.shape[-2]//patch_size)
    causal_mask = torch.zeros((image.shape[0],num_patches+num_sprites,num_patches+num_sprites))
    causal_mask[:,:num_patches,:num_patches] = torch.eye(num_patches).repeat(image.shape[0],1,1)
    sprite_mask_pre = get_sprites_in_patch(image,patch_size,num_sprites)
    sprite_mask_post = get_sprites_in_patch(next_image,patch_size,num_sprites)
    
    for b in range(next_image.shape[0]):
        for sprite_idx in range(num_sprites):
            sprite_patches_pre = torch.nonzero(sprite_mask_pre[b,sprite_idx])
            sprite_patches_post = torch.nonzero(sprite_mask_post[b,sprite_idx])
            # causal mask of states to states
            for patch_pre in sprite_patches_pre:
                for patch_post in sprite_patches_post:
                    causal_mask[b,patch_post,patch_pre] = 1
            
            # causal mask of actions to states
            for patch_post in sprite_patches_post:
                causal_mask[b,patch_post,num_patches+sprite_idx] = 1
            for patch_pre in sprite_patches_pre:
                causal_mask[b,patch_pre,num_patches+sprite_idx] = 1

    return causal_mask


def aggreg_heads(layer_attn, head_fusion):
    
    if head_fusion == "mean":
        attention_heads_fused = layer_attn.mean(axis=0)
    elif head_fusion == "max":
        attention_heads_fused = layer_attn.max(axis=0)[0]
    elif head_fusion == "min":
        attention_heads_fused = layer_attn.min(axis=0)[0]
    else:
        raise "Attention head fusion type Not supported"

    return attention_heads_fused

def postprocess_fused(attention_heads_fused,discard_ratio=0.98):
    
    _, indices = torch.topk(attention_heads_fused,int(attention_heads_fused.size(0)*discard_ratio), dim=0,largest= False)
    attention_heads_fused.scatter_(0,indices,0)# Set discarded weights to zero
    return attention_heads_fused
    

def attn_rollout(attentions, discard_ratio, head_fusion,residual=False,post_process=False):
    """
    attn_rollout Creates a causal mask from a sequence of attention maps
                    Only for batch size 1

    Args:
        attentions (_type_): attention weights of shape (num_layers, num_heads,seq_len,seq_len)
        discard_ratio (_type_): number of weights to discard
        head_fusion (_type_): how to fuse heads in same layer

    Returns:
        _type_: causal mask of shape (seq_len,seq_len)
    """

    result = torch.eye(attentions.size(-1)).to(attentions.device)
    
    with torch.no_grad():
        for attention in attentions:

            attention_heads_fused = aggreg_heads(attention, head_fusion)
            if post_process:    
                attention_heads_fused = postprocess_fused(attention_heads_fused,discard_ratio=discard_ratio)
            _, indices = torch.topk(attention_heads_fused,int(attention_heads_fused.size(-1)*discard_ratio), dim=-1,largest= False)
            attention_heads_fused.scatter_(-1,indices,0)# Set discarded weights to zero
            
            if residual:
                I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) # Add residual self attention
                a = (attention_heads_fused + 1.0*I)/2 
            else:
                a = attention_heads_fused
            #a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    return result    
