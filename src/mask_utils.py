from moog_demos.example_configs import bouncing_sprites
import torch
EPS = 1e-6


    
TARGET_POS_IMG = [[9,16,22,29],
                  [35,42,22,29],
                  [86,93,48,55],
                  [86,93,73,80]
                 ]

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
    image = remove_targets_from_image(image)
    sprite_colors = torch.Tensor(bouncing_sprites.color_list).to(image.device)
    sprites_in_patch = torch.zeros(( image.shape[0],num_sprites,(image.shape[2]//patch_size) * (image.shape[3]//patch_size)))
    for sprite_idx in range(num_sprites):
        sprite_color = sprite_colors[sprite_idx]
        sprite_image = ((image.permute(0,2,3,1) == torch.Tensor(sprite_color)).permute(0,3,1,2)).all(dim=1,keepdim=True).float()
        sprite_image_patches = image_to_image_patch_batch(sprite_image,patch_size=(patch_size,patch_size),num_patches_sqrt=(image.shape[2]//patch_size))
        sprite_image_patches = sprite_image_patches.sum((-3,-2,-1)) > 0
        sprite_image_patches = sprite_image_patches
        sprites_in_patch[:,sprite_idx] = sprite_image_patches
    sprites_in_patch = sprites_in_patch.flatten(-1)
    return sprites_in_patch    

def make_gt_causal_mask(images,actions,patch_size=16,num_sprites=4):
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
    num_patches = (images.shape[-1]//patch_size) * (images.shape[-2]//patch_size)
    causal_mask = torch.zeros((images.shape[0],num_patches+num_sprites,num_patches+num_sprites))
    causal_mask[:,:num_patches,:num_patches] = torch.eye(num_patches).repeat(images.shape[0],1,1)
    sprite_mask_pre = get_sprites_in_patch(images[:,0],patch_size,num_sprites)
    sprite_mask_post = get_sprites_in_patch(images[:,1],patch_size,num_sprites)
    
    for b in range(images.shape[0]):
        for sprite_idx in range(num_sprites):
            sprite_patches_pre = torch.nonzero(sprite_mask_pre[b,sprite_idx])
            sprite_patches_post = torch.nonzero(sprite_mask_post[b,sprite_idx])
            # causal mask of states to states
            for patch_pre in sprite_patches_pre:
                for patch_post in sprite_patches_post:
                    causal_mask[b,patch_post,patch_pre[0]] = 1
            
            # causal mask of actions to states
            for patch_post in sprite_patches_post:
                causal_mask[b,patch_post,num_patches+sprite_idx] = 1
        
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

def attn_rollout(attentions, discard_ratio, head_fusion,residual=False):
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

    result = torch.eye(attentions.size(-1))
    
    with torch.no_grad():
        for attention in attentions:

            attention_heads_fused = aggreg_heads(attention, head_fusion)
            _, indices = torch.topk(attention_heads_fused,int(attention_heads_fused.size(-1)*discard_ratio), dim=-1,largest= False)
            attention_heads_fused.scatter_(-1,indices,0)# Set discarded weights to zero
            
            if residual:
                I = torch.eye(attention_heads_fused.size(-1)) # Add residual self attention
                a = (attention_heads_fused + 1.0*I)/2 
            else:
                a = attention_heads_fused
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    return result    
