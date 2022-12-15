import gym
import torch 
import torch.nn as nn
import einops
import d3rlpy
from .networks.cnns import Conv2dModel


class PatchCNNEncoder(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_shape,feature_dim, patch_size=16):
        super(PatchCNNEncoder, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.cnn_downsample = Conv2dModel(in_channels=3, channels=[64,128,64,32],kernel_sizes=[5,3,3,3],strides=[2,2,2,2],paddings=[1,1,1,1],norm_type="gn")
        self.patch_size = patch_size
        # Compute shape by doing one forward pass
        self._feature_dim = feature_dim

        with torch.no_grad():
            n_flatten = self.cnns((torch.zeros((1,*observation_shape))).float())
            print(f"CNN output shape: {n_flatten.shape}")
        self.linear = nn.Sequential(nn.Linear(n_flatten.shape[1], feature_dim), nn.LeakyReLU())

    def cnns(self, observations: torch.Tensor) -> torch.Tensor:
        
        image_patches = einops.rearrange(observations, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
        patches_encodings = self.cnn_downsample(image_patches)
        img_patches_encodings = einops.rearrange(patches_encodings, '(b h w) c dh dw -> b c (h dh) (w dw) ',b=observations.shape[0], h=observations.shape[-1]//self.patch_size)
        return img_patches_encodings.flatten(start_dim=1)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        return self.linear(self.cnns(observations))

    def get_feature_size(self) -> int:
        return self._feature_dim

class PatchCNNEncoderWithAction(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_shape,action_size,feature_dim, patch_size=16):
        super(PatchCNNEncoderWithAction, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.cnn_downsample = Conv2dModel(in_channels=3, channels=[64,128,64,32],kernel_sizes=[5,3,3,3],strides=[2,2,2,2],paddings=[1,1,1,1],norm_type="gn")
        self.patch_size = patch_size
        self._action_size = action_size
        self._feature_dim = feature_dim
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnns(torch.zeros((1,*observation_shape)).float())
            print(f"CNN output shape: {n_flatten.shape}")
        self.linear = nn.Sequential(nn.Linear(n_flatten.shape[1], feature_dim), nn.ReLU())
        self.obs_action_linear = nn.Linear(feature_dim+action_size, feature_dim )
        self.out_act = nn.LeakyReLU()

    def cnns(self, observations: torch.Tensor) -> torch.Tensor:
        
        image_patches = einops.rearrange(observations, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
        patches_encodings = self.cnn_downsample(image_patches)
        img_patches_encodings = einops.rearrange(patches_encodings, '(b h w) c dh dw -> b c (h dh) (w dw) ',b=observations.shape[0], h=observations.shape[-1]//self.patch_size)
        return img_patches_encodings.flatten(start_dim=1)
        
    def forward(self, observations: torch.Tensor,action) -> torch.Tensor:
        
        h = self.linear(self.cnns(observations))
        h = torch.cat([h.view(h.shape[0], -1), action], dim=1)
        h = self.out_act(self.obs_action_linear(h))
        return h
    
    def get_feature_size(self) -> int:
        return self._feature_dim

    @property
    def action_size(self) -> int:
        return self._action_size

class PatchCNNFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"  # this is necessary

    def __init__(self, feature_dim, patch_size):
        self.feature_dim = feature_dim
        self.patch_size = patch_size

    def create(self, observation_shape):
        return PatchCNNEncoder(observation_shape, self.feature_dim,  self.patch_size)
    
    def create_with_action(
        self,
        observation_shape,
        action_size: int,
        discrete_action: bool = False,
    )  :
        assert not discrete_action, "Discrete action is not supported"
        return PatchCNNEncoderWithAction(observation_shape, action_size,self.feature_dim, self.patch_size)
    

    def get_params(self, deep=False):
        return {"feature_dim": self.feature_dim, "patch_size": self.patch_size}
