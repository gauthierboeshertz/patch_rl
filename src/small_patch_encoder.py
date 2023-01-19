import torch
import einops
from .patch_utils import  patches_to_image
from torch.nn.functional import one_hot

class SmallPatchEncoder():

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 colors,
                 patch_size:int = 32,
                 image_size:int = 128,
                 background_color = [0, 0, 0]
                 ) -> None:
        self.background_color = torch.tensor(background_color).unsqueeze(0)
        self.colors = torch.cat((self.background_color,torch.Tensor(colors)))
        self.embed_dim = self.colors.shape[0]  
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

    def get_encoding_for_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes the patches and returns the encodings as a tensor of shape (batch, num_patches, embed_dim)
        where the embeding is the one-hot encoding of the color 
        """
        patches = einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) p1 p2 c', p1=self.patch_size, p2=self.patch_size)

        num_pixel_of_color_per_patch = torch.zeros((patches.shape[0], self.num_patches, self.embed_dim))
        for i, color in enumerate(self.colors[1:]):
            i += 1
            num_pixel_of_color_per_patch[:, :, i] = (patches == color).all(-1).sum((-1,-2))
        max_color_per_patch = num_pixel_of_color_per_patch.argmax(-1)
        max_color_per_patch[ num_pixel_of_color_per_patch.sum(-1) == 0 ] = 0 
        max_color_per_patch = one_hot(max_color_per_patch, self.embed_dim).float()
        return max_color_per_patch        

    def decode(self,patches):
        colored_patches = torch.zeros((patches.shape[0],patches.shape[1],3))
        patches = patches.argmax(-1)
        for i, color in enumerate(self.colors):
            colored_patches[patches==i] = color
        print(colored_patches.shape)
        colored_patches = colored_patches.reshape((patches.shape[0],patches.shape[1],1,1,3))
        colored_patches = colored_patches.repeat(1,1,self.patch_size,self.patch_size,1)
        colored_patches = colored_patches.permute(0,1,4,2,3)
        print(colored_patches.shape)
        images = patches_to_image(colored_patches,patch_size=self.patch_size,image_size=self.image_size)#einops.rearrange(colored_patches, 'b (h w) p1 p2 c -> b c (h p1) (w p2)', h=(128//4),p1= int(spe.patch_size), p2=int(spe.patch_size))
        
        return images