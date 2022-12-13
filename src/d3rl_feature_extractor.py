import gym
import torch 
import torch.nn as nn
import einops
import d3rlpy



class PatchCNNEncoder(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_shape,feature_dim, cnn_downsample=None, patch_size=16):
        super(PatchCNNEncoder, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.cnn_downsample = cnn_downsample
        self.patch_size = patch_size
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnns(torch.as_tensor(observation_shape[None]).float())
            print(f"CNN output shape: {n_flatten.shape}")
        self.linear = nn.Sequential(nn.Linear(n_flatten.shape[1], feature_dim), nn.ReLU())

    def cnns(self, observations: torch.Tensor) -> torch.Tensor:
        
        image_patches = einops.rearrange(observations, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
        patches_encodings = self.cnn_downsample(image_patches)
        img_patches_encodings = einops.rearrange(patches_encodings, '(b h w) c dh dw -> b c (h dh) (w dw) ',b=observations.shape[0], h=observations.shape[-1]//self.patch_size)
        return img_patches_encodings.flatten(start_dim=1)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        return self.linear(self.cnns(observations))


class PatchCNNFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"  # this is necessary

    def __init__(self, feature_dim,cnn_downsample, patch_size):
        self.feature_dim = feature_dim
        self.cnn_downsample = cnn_downsample
        self.patch_size = patch_size

    def create(self, observation_shape):
        return PatchCNNEncoder(observation_shape, self.feature_dim, self.cnn_downsample, self.patch_size)

    def get_params(self, deep=False):
        return {"feature_dim": self.feature_dim, "cnn_downsample": self.cnn_downsample, "patch_size": self.patch_size}
