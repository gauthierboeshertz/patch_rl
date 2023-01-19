import torch
from torch import nn
from torch.nn import functional as F
import einops
from .patch_utils import  patches_to_image
from .networks.cnns import MLP
import numpy as np


class MLPPatchVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                in_channels: int,
                embed_dim: int,
                channels: list,
                categorical_dim: int = 40, # Num classes
                temperature: float = 0.5,
                anneal_rate: float = 3e-5,
                anneal_interval: int = 100, # every 100 batches
                alpha: float = 30,
                kld_weight= 0.001,
                patch_size:int = 32,
                norm_type="gn",
                name="") -> None:
        super(MLPPatchVAE, self).__init__()

        self.embed_dim = embed_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.min_temp = 0.01#temperature
        self.anneal_rate = anneal_rate
        self.kld_weight = kld_weight
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        
        self.patch_size = patch_size
        self.last_channel = channels[-1]        
        # Build Encoder
        input_dim = in_channels * patch_size * patch_size
        
        self.encoder = MLP(input_dim,channels[-1],layer_sizes=channels[:-1],activation=nn.LeakyReLU,norm_type=None)  #VAE_Encoder(in_channels, channels,kernel_sizes,strides,paddings,norm_type=norm_type)
        self.fc_z = nn.Linear(self.last_channel,
                               self.embed_dim * self.categorical_dim)

        channels.reverse()
        self.decoder = MLP(self.embed_dim * self.categorical_dim,input_dim,layer_sizes=channels[:-1],activation=nn.LeakyReLU,norm_type=norm_type)
        
        self.num_patches = self.encode(torch.zeros(1,3,128,128))[0].shape[0]

        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) List of latent codes
        """
        patches = self.images_to_patches(input)
        flat_patches = torch.flatten(patches, start_dim=1)
        result = self.encoder(flat_patches)
        result = torch.flatten(result, start_dim=1)
        #result = einops.rearrange(result," (b n) c -> b n c",b=input.shape[0])
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        z = z.view(-1, self.embed_dim, self.categorical_dim)
        return z

    def decode(self, z):
        #result = einops.rearrange(result," b  (c h w) -> b c h w ",h=2,w=2)#result = result.view(-1, self.last_channel, 2, 2)
        result = self.decoder(z)
        result = torch.sigmoid(result)
        result = einops.rearrange(result," b (c h w)  -> b c h w ",c=3,h=self.patch_size,w=self.patch_size)
        #result = einops.rearrange(result," b c h w  -> b n c h w ",b=z.shape[0])
        return result

    def reparameterize(self, z, eps:float = 1e-7) :
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)
        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.embed_dim * self.categorical_dim)
        return s

    def forward(self, input):
        """
        forward makes patches of an image, encode it  and decode it

        _extended_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        mu = self.encode(input)
        z = self.reparameterize(mu)
        
        return  [self.decode(z), z, mu]

    def get_encoding_for_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.encode(x)
        #mu = mu.argmax(dim=-1)
        mu = self.reparameterize(mu)
        mu = mu.view(-1, self.embed_dim*self.categorical_dim)
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
        
        
    def reconstruct_with_mu(self,images):
        mu = self.encode(images)
        decoded_patches = self.decode(mu)
        return einops.rearrange(decoded_patches,"(b h w) c p1 p2 -> b c (h p1) (w p2)",h=images.shape[-1]//self.patch_size ,b=images.shape[0],p1=self.patch_size,p2=self.patch_size)
        
    def loss_function(self,
                      recons,
                      input,
                      q) -> dict:
        self.num_iter += 1

        q_p = F.softmax(q, dim=-1) # Convert the categorical codes into probabilities

        kld_weight = self.kld_weight#kwargs['M_N'] # Account for the minibatch samples from the dataset
        batch_idx = self.num_iter#kwargs['batch_idx']

        # Anneal the temperature at regular intervals
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
                                   self.min_temp)
        recons_loss = F.mse_loss(recons, input, reduction='mean')

        # KL divergence between gumbel-softmax distribution
        eps = 1e-7

        # Entropy of the logits
        h1 = q_p * torch.log(q_p + eps)

        h2 = q_p * np.log(1. / self.categorical_dim + eps)
        kld_loss = torch.mean(torch.sum(h1 - h2, dim =(1,2)), dim=0)

        loss = self.alpha * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    
    def reconstruct_image(self,images):
        patches = self.images_to_patches(images)
        recons = self.forward(patches)[0]
        recons = einops.rearrange(recons, "(b n) c h w -> b n c h w", b=images.shape[0])
        recons_image = patches_to_image(recons,self.patch_size,images.shape[2])
        return recons_image

    def compute_loss_encodings(self, obs):
        ## obs should be of shape (batch, time ,channels, height, width)
        tobs = einops.rearrange(obs, "b t c h w -> (b t) c h w")
        recons, z, mu  = self(tobs)
        patches = einops.rearrange(tobs, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=self.patch_size, p2=self.patch_size)
        (recons, mu_removed), patches = self.remove_empty_patches_for_loss([recons, mu], patches) 
        loss_dict = self.loss_function(recons, patches, mu_removed)
        
        mu = F.softmax(mu, dim=-1)
        mu =  einops.rearrange(mu, "(b t n) c d  -> b t n (c d) ", b=obs.shape[0],t=obs.shape[1])
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
