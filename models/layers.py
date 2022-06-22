import flax.linen as nn


class FeedForward(nn.Module):
    """
    Simple feed forward module in the transformer architecture
        ==> x[..., latent] -> x[..., original]

    Input:
        x: Input tensor
        latent_dims: Latent dimensions of 2/3 fc layers in the architecture
            Use latent_dims(2) and latent_dims(3)

    """

    def apply(self, x, latent_dim):
        dim = x.shape[-1]  # Original input dimension of the patches
        x = nn.Dense(x, latent_dim)  # self.latent_dims[1]
        x = nn.gelu(x)
        x = nn.Dense(x, dim)
        return x


class Residual(nn.Module):
    """
    Residual network with input injection
        ==> x -> f(x) + x
    """

    def apply(self, x, f_res):
        return f_res(x) + x


class PreNorm(nn.Module):
    """
    Simple class for applying a function to normalized input
    """

    def apply(self, x, f):
        return f(nn.LayerNorm(x))


class Transformer(nn.Module):
    """
    Transformer
    """

    def __init__(self, depth, num_heads, latent_dims):
        self.depth = depth
        self.num_heads = num_heads
        self.latent_dims = latent_dims
        self.attention = nn.SelfAttention.partial(num_heads=self.num_heads)
        self.norm = nn.LayerNorm
        self.norm_attention = PreNorm.partial(
            norm=self.norm, f=self.attention)
        self.residual_norm_attention = Residual.partial(
            residual_fn=self.norm_attention)
        print("\n\n\nself.latent_dims: {}\n\n\n".format(self.latent_dims))
        self.forward = FeedForward.partial(latent_dim=self.latent_dims)
        self.forward_norm = PreNorm.partial(f=self.forward)
        self.forward_res = Residual.partial(f_res=self.forward_norm)

    def apply(self, x):
        # Create a module with ```self.depth layers```
        for _ in range(self.depth):
            # fc(gelu(fc(norm(SelfAttention(norm(x))))))
            x = self.forward_res(self.residual_norm_attention(x))
        print(x)

        return x
