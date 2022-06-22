import jax
import flax.linen as nn
import numpy as np

from models.layers import Transformer


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear', align_corners=True):
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.interp = jax.image.resize

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=(x.shape[0]//self.scale_factor,
                          x.shape[1]//self.scale_factor),
            mode=self.mode,
            align_corners=self.align_corners
        )
        return x


class HeadDepth(nn.Module):
    def __init__(self, features):
        self.features = features
        self.kernel_size = 2
        self.conv2d_1 = nn.Conv(self.features // 2,
                                kernel_size=self.kernel_size,
                                strides=1,
                                padding=1)
        self.conv2d_2 = nn.Conv(32,
                                kernel_size=self.kernel_size,
                                strides=1,
                                padding=1)
        self.conv2d_3 = nn.Conv(1,
                                kernel_size=1,
                                strides=1,
                                padding=0)
        self.interp = Interpolate(scale_factor=2)
        self.relu = nn.relu
        self.sigmoid = nn.sigmoid

    def apply(self, x):
        x = self.conv2d_1(x)
        x = self.interp(x)  # replace with transConv if necessary
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.conv2d_3(x)
        x = self.sigmoid(x)

        return x


class HeadSeg(nn.Module):
    def __init__(self, features, num_classes=2):
        self.features = features
        self.num_classes = num_classes
        self.kernel_size = 2
        self.conv2d_1 = nn.Conv(self.features // 2,
                                kernel_size=self.kernel_size,
                                strides=1,
                                padding=1)
        self.conv2d_2 = nn.Conv(32,
                                kernel_size=self.kernel_size,
                                strides=1,
                                padding=1)
        self.conv2d_3 = nn.Conv(self.num_classes,
                                kernel_size=1,
                                strides=1,
                                padding=0)
        self.interp = Interpolate(scale_factor=2)
        self.relu = nn.relu
        self.sigmoid = nn.sigmoid

    def apply(self, x):
        x = self.conv2d_1(x)
        x = self.interp(x)  # replace with transConv if necessary
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.conv2d_3(x)
        # x = self.sigmoid(x)
        return x

# TODO [PanSeg update incoming...]
# class HeadPanSeg(flax.nn.module):
# def __init__(self, features, num_classes):
#     self.features = features
#     self.num_classes = num_classes

# def forward(self, x):
#     x = self.conv2d_1(x)
#     x = self.interp(x)  # replace with transConv if necessary
#     x = self.conv2d_2(x)
#     x = self.relu(x)
#     x = self.conv2d_3(x)
#     x = self.sigmoid(x)
#     return x


class Params(nn.Module):
    init = nn.initializers.normal(stddev=1.0)

    @nn.compact
    def __call__(self, bsz, patches_qty, latent_dims):
        tokens_cls = self.param(
            'tokens_cls', (bsz, 1, latent_dims[1]), self.init)
        embed_pos = self.param(
            'embed_pos', (1, patches_qty+1, latent_dims[1]), self.init)
        return tokens_cls, embed_pos


class TransformerCV(nn.Module):
    def __init__(self,
                 x,
                 patch_size,
                 num_heads,
                 num_classes,
                 depth,
                 resample_dim,
                 mode,
                 latent_dims
                 ):
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth
        self.latent_dims = latent_dims
        self.mode = mode
        self.resample_dim = resample_dim
        self.head_depth = HeadDepth(self.resample_dim)
        self.head_seg = HeadSeg(self.resample_dim, self.num_classes)
        self.x = x
        self.fc = nn.Dense(self.latent_dims[0])
        bsz, hgt, wdt, cnl = self.x.shape
        # TODO: why was this axes_swap done?
        input = self.x.reshape(bsz, cnl, hgt, wdt)
        patches_qty = (hgt*wdt)//(self.patch_size *
                                  self.patch_size)
        patches_dim = cnl*self.patch_size**2
        patch = input.reshape(bsz, patches_qty, patches_dim)

        # Convert patch to fixed len embedding
        fc_vars = self.fc.init(jax.random.PRNGKey(0), patch)
        self.embed = self.fc.apply(fc_vars, patch)
        self.params = Params()
        fc_vars = self.params.init(None)
        self.tokens_cls, self.embed_pos = self.params.apply(fc_vars,
                                                            bsz, patches_qty, latent_dims)

    def __call__(self, x):
        """
        Apply the vision transformer to the given input (img tensor)

        Input:
            self.x: Input tensor image
            self.path_size: How many patches should the image be broken into?
            self.depth: How many residual layers do we want out transformer to have?
            self.num_heads: How many attensiotn heads? (Support choice of attention heads from literature)
            self.latent_dim: latent dim for fc layer

        Output:
            output_cls: output classification
            output seg: generated segmentation
        """

        # The embedding is incomplete, let's add the following:
        # - class tokens
        # - positional embeddings
        x = jax.lax.concatenate([self.tokens_cls, self.embed], axis=1)
        x += self.embed_pos
        x = Transformer(self.depth, self.num_heads,
                        self.latent_dims[1]).run(x)

        if (self.mode == "seg"):
            x = self.head_seg(x)
        elif (self.mode == "depth"):
            x = self.depth(x)
        elif (self.mode == "cls"):
            x = x[:, 0]
            x = nn.Dense(x, self.latent_dims[2])
            x = nn.gelu(x)
            x = nn.Dense(x, self.num_classes)
        elif (self.mode == "segdepth"):
            seg = self.head_seg(x)
            depth = self.depth(x)
            return seg, depth

        return x
