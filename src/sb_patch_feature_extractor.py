import gym
import torch 
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box
import einops
from .networks.cnns import Conv2dModel

class PatchVAEFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512,encoder_decoder=None):
        super(PatchVAEFeatureExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.encoder_decoder = encoder_decoder
        self.cnn_downsample = nn.Linear(self.encoder_decoder.embed_dim, 32)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnns(torch.as_tensor(observation_space.sample()[None]).float())
            print(f"CNN output shape: {n_flatten.shape}")
        self.linear = nn.Sequential(nn.Linear(n_flatten.shape[1], features_dim), nn.ReLU())

    def cnns(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            patches_encodings = self.encoder_decoder.get_encoding_for_dynamics(observations)
            #img_patches_encodings = einops.rearrange(patches_encodings, '(b h w) c -> b c h w',b=observations.shape[0], h=observations.shape[-1]//self.encoder_decoder.patch_size)
        downsampled_patches = self.cnn_downsample(patches_encodings)
        downsampled_patches = einops.rearrange(downsampled_patches, '(b h w) c -> b h w c',b=observations.shape[0], h=observations.shape[-1]//self.encoder_decoder.patch_size)
        return downsampled_patches.flatten(start_dim=1)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.linear(self.cnns(observations))


class PatchFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, patch_size=16):
        super(PatchFeatureExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.cnn_downsample =  Conv2dModel(in_channels=3, channels=[64,128,64,32],kernel_sizes=[5,3,3,3],strides=[2,2,2,2],paddings=[1,1,1,1],norm_type="gn")
        self.patch_size = patch_size
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnns(torch.as_tensor(observation_space.sample()[None]).float())
            print(f"CNN output shape: {n_flatten.shape}")
        self.linear = nn.Sequential(nn.Linear(n_flatten.shape[1], features_dim), nn.ReLU())

    def cnns(self, observations: torch.Tensor) -> torch.Tensor:
        
        image_patches = einops.rearrange(observations, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
        patches_encodings = self.cnn_downsample(image_patches)
        img_patches_encodings = einops.rearrange(patches_encodings, '(b h w) c dh dw -> b c (h dh) (w dw) ',b=observations.shape[0], h=observations.shape[-1]//self.patch_size)
        return img_patches_encodings.flatten(start_dim=1)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.linear(self.cnns(observations))
