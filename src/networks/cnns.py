from re import L
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, output_size,layer_sizes = [], activation=nn.ReLU):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.activation = activation
        print(f"Creating MLP with input size {input_size} and output size {output_size}")
        print(f"Creating MLP with layer sizes {layer_sizes}")
        self.layers = []
        if len(layer_sizes) == 0:
            self.layers = nn.Linear(input_size, output_size)
        else:
            self.layers.append(nn.Linear(input_size, layer_sizes[0]))
            for i in range(len(layer_sizes)-1):
                self.layers.append(self.activation())
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(self.activation())
            self.layers.append(nn.Linear(layer_sizes[-1], output_size))
            
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)

def fixup_init(layer, num_layers):
    nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 norm_type, num_layers=1, groups=-1,
                 drop_prob=0., bias=True):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2, 3]
        self.drop_prob = drop_prob

        hidden_dim = round(in_channels * expand_ratio)

        if groups <= 0:
            groups = hidden_dim

        conv = nn.Conv2d

        if stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, stride, stride)
            nn.init.normal_(self.downsample.weight, mean=0, std=
                            np.sqrt(2 / (self.downsample.weight.shape[0] *
                            np.prod(self.downsample.weight.shape[2:]))))
        else:
            self.downsample = False

        if expand_ratio == 1:
            conv1 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv2 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            self.conv = nn.Sequential(
                # dw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv2,
                init_normalization(out_channels, norm_type),
            )
            nn.init.constant_(self.conv[-1].weight, 0)
        else:
            conv1 = conv(in_channels, hidden_dim, 1, 1, 0, bias=bias)
            conv2 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv3 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            fixup_init(conv3, num_layers)
            self.conv = nn.Sequential(
                # pw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # dw
                conv2,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv3,
                init_normalization(out_channels, norm_type)
            )
            if norm_type != "none":
                nn.init.constant_(self.conv[-1].weight, 0)

    def forward(self, x):
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x
        if self.training and np.random.uniform() < self.drop_prob:
            return identity
        else:
            return identity + self.conv(x)


class Residual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, groups=1)


class ResnetCNN(nn.Module):
    def __init__(self, input_channels,
                 depths=(16, 32, 64),
                 strides=(3, 2, 2),
                 blocks_per_group=3,
                 norm_type="bn",
                 resblock=InvertedResidual,
                 expand_ratio=2,):
        super(ResnetCNN, self).__init__()
        self.depths = [input_channels] + depths
        self.resblock = resblock
        self.expand_ratio = expand_ratio
        self.blocks_per_group = blocks_per_group
        self.layers = []
        self.norm_type = norm_type
        self.num_layers = self.blocks_per_group*len(depths)
        for i in range(len(depths)):
            self.layers.append(self._make_layer(self.depths[i],
                                                self.depths[i+1],
                                                strides[i],
                                                ))
        self.layers = nn.Sequential(*self.layers)
        self.train()

    def _make_layer(self, in_channels, depth, stride,):

        blocks = [self.resblock(in_channels, depth,
                                expand_ratio=self.expand_ratio,
                                stride=stride,
                                norm_type=self.norm_type,
                                num_layers=self.num_layers,)]

        for i in range(1, self.blocks_per_group):
            blocks.append(self.resblock(depth, depth,
                                        expand_ratio=self.expand_ratio,
                                        stride=1,
                                        norm_type=self.norm_type,
                                        num_layers=self.num_layers,))

        return nn.Sequential(*blocks)

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs):
        return self.layers(inputs)




def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "gn", "max", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "gn":
        groups = max(min(32, channels//4), 1)
        return nn.GroupNorm(groups, channels, affine=affine)
    elif type == "max":
        if not one_d:
            return renormalize
        else:
            return lambda x: renormalize(x, -1)
    elif type == "none" or type is None:
        return nn.Identity()


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU,  # Module, not Functional.
            dropout=0.,
            norm_type="none",
            use_maxpool=False
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride, oc in zip(conv_layers, maxp_strides, channels):
            sequence.extend([conv_layer, init_normalization(oc, norm_type), nonlinearity()])
            #sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        
        return self.conv(input)

class DeConv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.ConvTransposes2d``.
    Only used for image generation of channel size 3.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU,  # Module, not Functional.
            dropout=0.,
            norm_type="none",
            use_maxpool=False
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.ConvTranspose2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride, oc in zip(conv_layers, maxp_strides, channels):
            sequence.extend([conv_layer, init_normalization(oc, norm_type), nonlinearity()])
            #sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        sequence.append(
            nn.Conv2d(channels[-1], out_channels= 3,
                kernel_size= 3, padding= 1),
            nn.Tanh())
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        
        return self.conv(input)


class PatchConv2dModel(torch.nn.Module):
    """2-D Convolutional model component,
        stride will be equql to the kernel size such that eatch patch will be independent,
        no two patches will be connected through the convolutional layers.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            dropout=0.,
            norm_type="bn",
            ):
        super().__init__()
        
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) ==  len(paddings)
        in_channels = [in_channels] + channels[:-1]
        
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=k, padding=p) for (ic, oc, k,  p) in
            zip(in_channels, channels, kernel_sizes, paddings)]
        sequence = list()
        
        for conv_layer, oc in zip(conv_layers, channels):
            sequence.extend([conv_layer, init_normalization(oc, norm_type), nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        
        return self.conv(input)

def GroupNorm(in_channels: int) -> nn.Module:
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class VAE_Encoder(nn.Module):
    
    def __init__(self,
        in_channels,
        channels,
        kernel_sizes,
        strides,
        paddings,
        norm_type="bn"):
        super().__init__()
        modules = []
        in_chan = in_channels
        
        if norm_type == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "gn":
            norm_layer = GroupNorm
        else:
            raise ValueError("Invalid norm_type: {}".format(norm_type))
        
        for chan, kern, stri,padd, in zip(channels,kernel_sizes,strides,paddings):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_chan, out_channels=chan,
                              kernel_size=kern,stride=stri,
                              padding=padd,bias=False),#bias set to false since batcchnorm does  input - mean(input)
                               # kernel_size= 3, stride= 2, padding  = 1),
                    norm_layer(chan),
                    nn.LeakyReLU())
            )
            in_chan = chan

            
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)
        return x


class VAE_Decoder(nn.Module):
    def __init__(self,
                channels,
                kernel_sizes,
                strides,
                paddings,
                norm_type="bn",
                out_dim=-1):
        super().__init__()

        modules = []
        in_chan = channels[0]

        if norm_type == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "gn":
            norm_layer = GroupNorm
        else:
            raise ValueError("Invalid norm_type: {}".format(norm_type))

        for chan, kern, stri,padd, in zip(channels,kernel_sizes,strides,paddings):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_chan,
                                       chan,
                                       kernel_size=kern,
                                       stride = stri,
                                       padding=padd,
                                       output_padding=1,
                                       bias=False),
                    norm_layer(chan),
                    nn.LeakyReLU())
            )
            in_chan = chan


        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(channels[-1],
                                               channels[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1,
                                               bias=False),
                            nn.BatchNorm2d(channels[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(channels[-1], out_channels=out_dim,
                                      kernel_size= 3, padding= 1,stride=2),
                            nn.Sigmoid())

        """
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(channels[-1],
                                               channels[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1,
                                               bias=False),
                            nn.BatchNorm2d(channels[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(channels[-1], out_channels=out_dim,
                                      kernel_size= 3, padding= 2,stride=3),
                            nn.Sigmoid())
        """
        
    def forward(self, x):
        x = self.decoder(x)
        x = self.final_layer(x)
        return x
    
class VAE_Encoder3D(nn.Module):
    
    def __init__(self,
        in_channels,
        channels,
        kernel_sizes,
        strides,
        paddings):
        super().__init__()
        modules = []
        in_chan = in_channels
        for chan, kern, stri,padd, in zip(channels,kernel_sizes,strides,paddings):
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_chan, out_channels=chan,
                              kernel_size=kern,stride=stri,
                              padding=padd,bias=False),#bias set to false since batcchnorm does  input - mean(input)
                               # kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm3d(chan),
                    nn.LeakyReLU())
            )
            in_chan = chan

            
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)
        return x

class VAE_Decoder3D(nn.Module):
    def __init__(self,
                channels,
                out_dim=3):
        super().__init__()

        modules = []
        in_chan = channels[0]
        convs = []
        convs.append(nn.Sequential(nn.ConvTranspose3d(in_chan,
                                            channels[1],
                                            kernel_size=(3, 3, 3),
                                            stride = 2,
                                            padding=1,
                                            output_padding=1,
                                            bias=False),
                    nn.BatchNorm3d(channels[1]),
                    nn.LeakyReLU()))

        convs.append(nn.Sequential(nn.ConvTranspose3d(channels[1],
                                            channels[2],
                                            kernel_size=(3, 3, 3),
                                            stride = 2,
                                            padding=1,
                                            output_padding=1,
                                            bias=False),
                    nn.BatchNorm3d(channels[2]),
                    nn.LeakyReLU()))

        convs.append(nn.Sequential(nn.ConvTranspose3d(channels[2],
                                            channels[3],
                                            kernel_size=(3, 3, 3),
                                            stride = 2,
                                            padding=1,
                                            output_padding=1,
                                            bias=False),
                    nn.BatchNorm3d(channels[3]),
                    nn.LeakyReLU()))

        convs.append(nn.Sequential(nn.ConvTranspose3d(channels[3],
                                            channels[4],
                                            kernel_size=(1, 3, 3),
                                            stride = (1,2,2),
                                            padding=(0,1,1),
                                            output_padding=(0,1,1),
                                            bias=False),
                    nn.BatchNorm3d(channels[4]),
                    nn.LeakyReLU()))


        convs.append(nn.Sequential(nn.Conv3d(channels[4],
                                            channels[5],
                                            kernel_size=(3, 1, 1),
                                            stride = (3,1,1),
                                            padding=(1,0,0),
                                            bias=False),
                    nn.BatchNorm3d(channels[5]),
                    nn.LeakyReLU()))

        convs.append(nn.Sequential(nn.Conv3d(channels[5],
                                            out_dim,
                                            kernel_size=(3, 1, 1),
                                            stride = (3,1,1),
                                            padding=(1,0,0),
                                            bias=False),
                     nn.Sigmoid()))
        self.decoder = nn.Sequential(*convs)
    def forward(self, x):
        
        return self.decoder(x)