import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp

from models.layers import Transformer


class Interpolate(hk.Module):
    def __init__(self, scale_factor, mode='bilinear', align_corners=True):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.method = mode
        self.align_corners = align_corners
        self.interp = jax.image.resize

    def __call__(self, x):
        print(self.scale_factor)
        x = self.interp(
            x,
            shape=(x.shape[0]*self.scale_factor,
                   x.shape[1]*self.scale_factor,
                   x.shape[2]),
            method=self.method,
        )
        return x


class HeadDepth(hk.Module):
    def __init__(self, features):
        super(HeadDepth, self).__init__()
        self.features = features
        self.kernel_size = 2
        self.conv2d_1 = hk.Conv2D(self.features // 2,
                                  kernel_shape=self.kernel_size,
                                  stride=1,
                                  padding=[1, 1])
        self.conv2d_2 = hk.Conv2D(32,
                                  kernel_shape=self.kernel_size,
                                  stride=1,
                                  padding=[1, 1])
        self.conv2d_3 = hk.Conv2D(1,
                                  kernel_shape=1,
                                  stride=1,
                                  padding=[0, 0])
        self.interp = Interpolate(scale_factor=2)
        self.relu = jax.nn.relu
        self.sigmoid = jax.nn.sigmoid

    def __call__(self, x):
        print("x.shape: {}".format(x.shape))
        x = self.conv2d_1(x)
        x = self.interp(x)  # replace with transConv?
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.conv2d_3(x)
        x = self.sigmoid(x)

        return x


class HeadSeg(hk.Module):
    def __init__(self, features, num_classes=2):
        super(HeadSeg, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.kernel_size = 2
        self.conv2d_1 = hk.Conv2D(self.features // 2,
                                  kernel_shape=self.kernel_size,
                                  stride=1,
                                  padding=[1, 1])
        self.conv2d_2 = hk.Conv2D(32,
                                  kernel_shape=self.kernel_size,
                                  stride=1,
                                  padding=[1, 1])
        self.conv2d_3 = hk.Conv2D(self.num_classes,
                                  kernel_shape=1,
                                  stride=1,
                                  padding=[0, 0])
        self.interp = Interpolate(scale_factor=2)
        self.relu = jax.nn.relu
        self.sigmoid = jax.nn.sigmoid

    def __call__(self, x):
        print("x.shape: {}".format(x.shape))
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


class TransformerCV(hk.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_heads,
                 num_classes,
                 depth,
                 resample_dim,
                 mode,
                 latent_dims
                 ):
        super(TransformerCV, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth
        self.latent_dims = latent_dims
        self.mode = mode
        self.init = jax.nn.initializers.normal(stddev=1.0)
        self.resample_dim = resample_dim
        self.head_depth = HeadDepth(self.resample_dim)
        self.head_seg = HeadSeg(self.resample_dim, self.num_classes)
        self.fc = hk.Linear(self.latent_dims[0])
        self.bsz, self.hgt, self.wdt, self.cnl = self.image_size
        self.patches_qty = (self.hgt*self.wdt)//(self.patch_size *
                                                 self.patch_size)
        self.patches_dim = self.cnl*self.patch_size**2
        self.tokens_cls = hk.get_parameter(
            'tokens_cls', shape=(self.bsz, 1, self.latent_dims[1]), init=jnp.zeros)
        self.embed_pos = hk.get_parameter(
            'embed_pos', shape=(1, self.patches_qty+1, self.latent_dims[1]), init=jnp.zeros)

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
        input = x.reshape(self.bsz, self.cnl, self.hgt, self.wdt)
        self.patches_qty = (self.hgt*self.wdt)//(self.patch_size *
                                                 self.patch_size)
        self.patches_dim = self.cnl*self.patch_size**2
        patch = input.reshape(self.bsz, self.patches_qty, self.patches_dim)

        # Convert patch to fixed len embedding
        embed = self.fc(patch)
        x = jnp.concatenate([self.tokens_cls, embed], axis=1)
        x += self.embed_pos
        x = Transformer(self.depth, self.num_heads,
                        self.latent_dims[1])(x)

        if (self.mode == "seg"):
            x = self.head_seg(x)
        elif (self.mode == "depth"):
            x = self.head_depth(x)
        elif (self.mode == "cls"):
            x = x[:, 0]
            x = hk.Linear(self.latent_dims[2])(x)
            x = jax.nn.gelu(x)
            x = hk.Linear(self.num_classes)(x)
        elif (self.mode == "segdepth"):
            seg = self.head_seg(x)
            depth = self.depth(x)
            return seg, depth

        return x
