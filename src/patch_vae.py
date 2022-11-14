from cmath import log
import torch
from torch import nn
from torch.nn import functional as F
from .networks.cnns import VAE_Encoder, VAE_Decoder
import einops
from .patch_utils import  patches_to_image

class PatchVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
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
                 patch_size:int = 32,
                 name="") -> None:
        super(PatchVAE, self).__init__()

        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.patch_size = patch_size
        self.kld_weight = kld_weight
        self.last_channel = channels[-1]        
        # Build Encoder
        self.encoder = VAE_Encoder(in_channels, channels,kernel_sizes,strides,paddings)
        self.fc_mu = nn.Linear(channels[-1]*4, embed_dim)
        self.fc_var = nn.Linear(channels[-1]*4, embed_dim)

        # Build Decoder
        
        self.decoder_input = nn.Linear(embed_dim, channels[-1] * 4)

        channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()
        self.decoder = VAE_Decoder(channels,kernel_sizes,strides,paddings,out_dim=in_channels)
        
        
        self.num_patches = self.encode(torch.zeros(1,3,128,128))[0].shape[0]

        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) List of latent codes
        """
        patches = self.images_to_patches(input)
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
        result = einops.rearrange(result," b  (c h w) -> b c h w ",h=2,w=2)#result = result.view(-1, self.last_channel, 2, 2)
        result = self.decoder(result)
        #result = einops.rearrange(result," b c h w  -> b n c h w ",b=z.shape[0])
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

    def get_encoding_for_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x)
        return mu#self.reparameterize(mu, log_var)


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
    
    def reconstruct_image(self,images):
        patches = self.images_to_patches(images)
        recons = self.forward(patches)[0]
        recons = einops.rearrange(recons, "(b n) c h w -> b n c h w", b=images.shape[0])
        recons_image = patches_to_image(recons,self.patch_size,images.shape[2])
        return recons_image

    def compute_loss_encodings(self, obs):
        ## obs should be of shape (batch, time ,channels, height, width)
        tobs = einops.rearrange(obs, "b t c h w -> (b t) c h w")
        recons, z, mu, log_var = self(tobs)
        patches = einops.rearrange(tobs, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=self.patch_size, p2=self.patch_size)
        #(recons, mu_removed, log_var), patches = self.remove_empty_patches_for_loss([recons, mu, log_var], patches) 
        loss_dict = self.loss_function(recons, patches, mu, log_var)
        
        mu =  einops.rearrange(mu, "(b t n) c -> b t n c", b=obs.shape[0],t=obs.shape[1])
        return loss_dict["loss"], mu, loss_dict
    
    def remove_empty_patches_for_loss(self, prediction, target):
        #remove patches that are all zeros
        # patches is of shape (b c h w)
        empty_patches = (target.sum(axis=(1,2,3)) == 0)
        target  = target[~empty_patches]
        if isinstance(prediction,list):
            prediction = [pred[~empty_patches] for pred in prediction]
        else:
            prediction = prediction[~empty_patches]
        return prediction, target
