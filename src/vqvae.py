"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import copy
import hydra
from .networks.vq_cnns import Encoder, Decoder,EncoderDecoderConfig
import einops
from .patch_utils import  patches_to_image
@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class VQVAE(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder, decoder,in_channels=3,name="") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        encoder_conf = EncoderDecoderConfig(**encoder)
        self.encoder = Encoder(encoder_conf)
        self.pre_quant_conv = torch.nn.Conv2d(encoder.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        decoder_conf = encoder_conf
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder_conf.z_channels, 1)
        self.decoder = Decoder(decoder_conf)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        
        self.num_patches = self.get_encoding_for_dynamics(torch.zeros(1,3,128,128)).shape[1]
        
        
    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, outputs.tokens, reconstructions

    def compute_loss_encodings(self, obs) :
        # obs is of shape (B,  T, C, H, W)
        t_obs = einops.rearrange(obs, 'b t c h w -> (b t) c h w')
        z, z_quantized, tokens,reconstructions = self(t_obs, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(t_obs - reconstructions).mean()

        tokens = einops.rearrange(tokens, '(b t) c -> b t c',b=obs.shape[0])
        return commitment_loss + reconstruction_loss, tokens# normaly has a 1 for the time dimension in shape[1]

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        
        print("encoder output",z.shape)
        
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        print("decoder input",z_q.shape)
        rec = self.decoder(z_q)
        print("decoder output",rec.shape)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec
    
    def reconstruct_image(self,images):
        return self(images, should_preprocess=False, should_postprocess=False)[-1]


    def get_encoding_for_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x, should_preprocess=False).tokens
    
    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
