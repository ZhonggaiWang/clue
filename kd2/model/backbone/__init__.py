from functools import partial
from torch import nn
from .vit import VisionTransformer


def deit_base_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError("Use --pretrained_weights for explicit local ImageNet initialization")
    return VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                             mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


vit_base_patch16_224 = deit_base_patch16_224
