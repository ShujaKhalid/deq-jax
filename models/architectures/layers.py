import jax
import haiku as hk
import jax.numpy as jnp
from einops import rearrange
from functools import partial


class FeedForward(hk.Module):
    """
    Simple feed forward module in the transformer architecture
        ==> x[..., latent] -> x[..., original]

    Input:
        x: Input tensor
        latent_dims: Latent dimensions of 2/3 fc layers in the architecture
            Use latent_dims(2) and latent_dims(3)

    """

    def __init__(self, latent_dim):
        super(FeedForward, self).__init__()
        self.latent_dim = latent_dim

    def __call__(self, x):
        dim = x.shape[-1]  # Original input dimension of the patches
        x = hk.Linear(self.latent_dim)(x)  # self.latent_dims[1]
        x = jax.nn.gelu(x)
        x = hk.Linear(dim)(x)
        return x


class Residual(hk.Module):
    """
    Residual network with input injection
        ==> x -> f(x) + x
    """

    def __init__(self, f_res):
        super(Residual, self).__init__()
        self.f_res = f_res

    def __call__(self, x):
        return self.f_res(x) + x


class PreNorm(hk.Module):
    """
    Simple class for applying a function to normalized input
    """

    def __init__(self, f):
        super(PreNorm, self).__init__()
        self.f = f

    def __call__(self, x):
        return self.f(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x))


class IdentityLayer(hk.Module):
    """
    Return the equivalent of an idedntity layer
    """

    def __call__(self, x):
        x = hk.Sequential([])
        return x


class SelfAttention(hk.Module):
    def __init__(self, dim_out, heads, dim_head=64, dropout=0):
        super(SelfAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim_out)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = hk.Linear(output_size=inner_dim * 3, with_bias=False)
        self.to_out = hk.Sequential([
            hk.Linear(dim_out),
            # hk.dropout(dropout, rate=0.2)  # TODO: add to config
        ]) if project_out else IdentityLayer()

    def __call__(self, x):
        # print("x.shape (before to_qkv): {}".format(x.shape))
        qkv = self.to_qkv(x)
        # print("qkv.shape (after to_qkv): {}".format(qkv.shape))
        qkv = jnp.split(qkv, 3, axis=-1)
        #print("qkv.shape (before rearrange): {}".format(qkv.shape))
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # print("q.shape (after map): {}".format(q.shape))
        # print("k.shape (after map): {}".format(k.shape))
        # print("v.shape (after map): {}".format(v.shape))

        # TODO: got this working but the paper implementation is different
        dots = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = jax.nn.softmax(dots, axis=-1)
        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
        # TODO: Is this really needed in VisTransformer archs?
        out = rearrange(out,  'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(hk.Module):
    """
    Transformer
    """

    def __init__(self, depth, num_heads, latent_dims):
        super(Transformer, self).__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.latent_dims = latent_dims
        self.attention = SelfAttention(
            dim_out=latent_dims, heads=self.num_heads)
        self.norm_attention = PreNorm(f=self.attention)
        self.residual_norm_attention = Residual(f_res=self.norm_attention)

        self.forward = FeedForward(latent_dim=self.latent_dims)
        self.forward_norm = PreNorm(f=self.forward)
        self.forward_res = Residual(f_res=self.forward_norm)

    def __call__(self, x):
        # Create a module with ```self.depth layers```
        for _ in range(self.depth):
            # fc(gelu(fc(norm(SelfAttention(norm(x))))))
            x = self.residual_norm_attention(x)
            # print("x.shape before res: {}".format(x.shape))
            x = self.forward_res(x)
            # print("x.shape after res: {}".format(x.shape))
        return x
