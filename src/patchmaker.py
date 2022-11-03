from unittest.mock import patch
import torch
import torch.nn as nn
import hydra
import einops

class PatchMaker(nn.Module):
    
    def __init__(self, encoder, patch_size=16):
        super(PatchMaker,self).__init__()
        assert patch_size % 2 == 0, "Patch size must be even"
        self.patch_size = patch_size
        self.encoder = hydra.utils.instantiate(encoder)
        
        
    def forward(self,x):
        """
        forward  makes patches then encodes them

        Args:
            x (_type_): batch of images shape (batch_size, channels, height, width)

        Returns:
            _type_: tensor of shape (batch_size, num_patches, num_channels)
        """        
        patches = self.images_to_patches(x)
        #patches = patches.view(-1,x.shape[1],self.patch_size,self.patch_size)
        patches = self.encoder(patches).squeeze()
        patches = einops.rearrange(patches, '(b n) c -> b n c', b=x.shape[0])
        return patches.contiguous()

    def images_to_patches(self,images):
        """
        image_to_patch Make an image as a list of patches of size (patch_size,patch_size)

        Args:
            images (_type_): image batch of shape (batch, channels, height, width)

        Returns:
            _type_: batch of list of patches of shape (batch, num_patches, channels, patch_size, patch_size)
        """        
        patches = einops.rearrange(images,"b c (h p1) (w p2) -> (b h w) c p1 p2",p1=self.patch_size,p2=self.patch_size)
        return patches
