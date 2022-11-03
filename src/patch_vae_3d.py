from cmath import log
import torch
from torch import nn
from torch.nn import functional as F
from .networks.cnns import VAE_Encoder3D, VAE_Decoder3D
import einops

class PatchVAE3D(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 kld_weight: float = 0.005,
                 patch_size:int = 32) -> None:
        super(PatchVAE3D, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.patch_size = patch_size
        self.kld_weight = kld_weight
        self.last_channel = channels[-1]        
        # Build Encoder
        
        self.encoder = VAE_Encoder3D(in_channels, channels,kernel_sizes,strides,paddings)
        self.fc_mu = nn.Linear(channels[-1]*4*2, latent_dim)
        self.fc_var = nn.Linear(channels[-1]*4*2, latent_dim)

        # Build Decoder
        
        self.decoder_input = nn.Linear(latent_dim, channels[-1] * 4 * 3)

        dec_channels = [channels[-1],128,128,64,64,32]
        self.decoder = VAE_Decoder3D(dec_channels,out_dim=3)
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        patches = self.images_to_patches(input)
        torch.save(patches,"patches.pt")
        result = self.encoder(patches)
        result = torch.flatten(result, start_dim=1)
        #result = einops.rearrange(result," (b n) c -> b n c",b=input.shape[0])
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = einops.rearrange(result," b  (c t h w) -> b c t h w ",c=self.latent_dim,h=2,w=2)#result = result.view(-1, self.last_channel, 2, 2)
        result = self.decoder(result)
        #result = einops.rearrange(result," b c h w  -> b n c h w ",b=z.shape[0])
        result = einops.rearrange(result,"  b c t h w  -> b (t c) h w ")
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        """
        forward makes patches of an image, encode it  and decode it

        _extended_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        
        return  [self.decode(z), z, mu, log_var]

    def get_embeddings(self,input):
        mu, log_var = self.encode(input)
        return self.reparameterize(mu, log_var)

    def images_to_patches(self,images):
        """
        image_to_patch Make an image as a list of patches of size (patch_size,patch_size)

        Args:
            images (_type_): image batch of shape (batch, channels, height, width)

        Returns:
            _type_: batch of list of patches of shape (batch, num_patches, channels, patch_size, patch_size)
        """        
        patches = einops.rearrange(images,"b (t c) (h p1) (w p2) -> (b h w) t c p1 p2",c=3,p1=self.patch_size,p2=self.patch_size)
        return patches
        
    
    def loss_function(self,
                      recons,
                      input,
                      mu,
                      log_var) -> dict:
        self.num_iter += 1

        recons_loss = F.mse_loss(recons, input)
                    
            
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * self.kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * self.kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}
