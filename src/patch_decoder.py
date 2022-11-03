import torch
import torch.nn as nn
import hydra
import einops
import math

class PatchDecoder(nn.Module):
    
    def __init__(self, decoder, patch_size=16):
        super(PatchDecoder,self).__init__()
        self.patch_size = patch_size
        self.decoder_input = nn.Linear(decoder["channels"][0], decoder["channels"][0]*4)
        self.decoder = hydra.utils.instantiate(decoder)
        
        
    def forward(self,x):
        """
        forward makes patches from the input image, either makes patches then encodes them or encodes then makes patches

        Args:
            x (_type_): batch of patches of  shape (batch_size, num_patches, patches_dim)

        Returns:
            _type_: tensor of shape (batch_size, channel, h, w)
        """        
        B, N, C = x.shape
        patches = einops.rearrange(x, 'b n c -> (b n) c')
        patches = self.decoder_input(patches)
        patches = patches.view(patches.shape[0],C,2,2)
        patches = self.decoder(patches)
        patches = einops.rearrange(patches, '(b n) c p1 p2 -> b n c p1 p2',b= x.shape[0], p1=self.patch_size, p2=self.patch_size)
        return patches.contiguous()