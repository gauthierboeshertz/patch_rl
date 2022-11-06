import math
import random
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    RandomResizedCrop,ColorJiggle,RandomGaussianNoise,RandomSharpness,VideoSequential
    
    

from kornia.filters import GaussianBlur2d
EPS = 1e-6

def plot_image_patches(image,patch_size,num_patches_sqrt):
    """ 
    Divide the image into patches of size patch_size and plot them
    """
    def image_to_image_patch(image,patch_size):
        image_patches = []
        for i in range(num_patches_sqrt):
            for j in range(num_patches_sqrt):
                image_patches.append(image[:,i*patch_size[0]:(i+1)*patch_size[0],j*patch_size[1]:(j+1)*patch_size[1]])
        return torch.stack(image_patches)
    img_patches = image_to_image_patch(image,patch_size).byte()
    img_patches = img_patches.permute(0,2,3,1)
    fig = plt.figure(figsize=(9, 13))
    columns = num_patches_sqrt
    rows = num_patches_sqrt
    num_img_seq = img_patches.shape[-1]//3
    # ax enables access to manipulate each of subplots
    ax = []

    print(img_patches.shape)
    for i in range(columns*rows):
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title("patch:"+str(i))  # set title
        if num_img_seq == 1:
            plt.imshow(img_patches[i])
        else:
            for t in range(num_img_seq):
                plt.imshow(img_patches[i,:,:,t*3:(t+1)*3],alpha=0.2*(t+1))
            
    plt.show()
    return img_patches

def renormalize(tensor, first_dim=-2):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
#    print("MIN",min)
#    print("MAX",max)
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)



def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def chain(*iterables):
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_((1-tau) * param.data +
                                tau * target_param.data)


def hard_update_params(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()



class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=tensor.device), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def get_augmentation(augmentation, imagesize,aug_prob):
    if isinstance(augmentation, str):
        augmentation = augmentation.split("_")
    transforms = []
    for aug in augmentation:
        if aug == "affine":
            transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5), p=aug_prob)
        elif aug == "rrc":
            transformation = RandomResizedCrop((imagesize, imagesize), (0.8, 1), p=aug_prob)
        elif aug == "blur":
            transformation = GaussianBlur2d((3, 3), (1., 1.))
        elif aug == "shift" or aug == "crop":
            transformation = nn.Sequential(nn.ReplicationPad2d(20), RandomCrop((imagesize, imagesize)), p=aug_prob)
        elif aug == "intensity":
            transformation = Intensity(scale=0.05)
        elif aug == "noise":
            transformation = RandomGaussianNoise(mean=0., std=0.002, p=aug_prob)
        elif aug == "sharp":
            transformation = RandomSharpness(sharpness=0.5, p=aug_prob)
        elif aug == "color":
            transformation = ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0., p=aug_prob)

        elif aug == "none":
            continue
        else:
            raise NotImplementedError()
        transforms.append(transformation)

    return VideoSequential(*transforms, data_format="BTCHW", same_on_frame=True)


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


def maybe_transform(image, transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * image
        return processed_images

