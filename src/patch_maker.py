import torch
import torch.nn as nn
from .networks.cnns import PatchConv2dModel
import hydra
import einops

class PatchMaker(nn.Module):
    
    def __init__(self, encoder_dict, encode_then_patch, patch_size=16):
        super(PatchMaker,self).__init__()
        self.encode_then_patch = encode_then_patch
        self.patch_size = patch_size
        self.encoder = hydra.utils.instantiate(encoder_dict)
        
        
    def forward(self,x):
        """
        forward makes patches from the input image, either makes patches then encodes them or encodes then makes patches

        Args:
            x (_type_): batch of images

        Returns:
            _type_: tensor of shape (batch_size, num_patches, num_channels)
        """        
        if self.encode_then_patch:
            feature_map = self.encoder(x)
            patches = einops.rearrange(feature_map, 'b c h w -> b (h w) c')
            
        else:
            patches = einops.rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
            patches = self.encoder(patches)
        return patches.contiguous()