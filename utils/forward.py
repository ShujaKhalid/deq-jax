import jax.numpy as jnp
import haiku as hk

from typing import Mapping
from models import resnet

from utils.utils import run


class Forward():
    def __init__(self, config):
        self.config = config
        self.mode = self.config["mode"]
        self.deq_flag = self.config["deq_attrs"]["deq_flag"]
        self.solver = self.config["deq_attrs"]["solver"]
        self.num_classes = self.config["data_attrs"]["num_classes"]
        self.batch_size = config["model_attrs"]["batch_size"]

        if (self.mode == "seg"):
            self.num_layers = self.config["model_attrs"]["cv"]["num_layers"]
            self.dropout_rate = self.config["model_attrs"]["cv"]["dropout_rate"]
            self.patch_size = self.config["model_attrs"]["cv"]["patch_size"]
            self.num_heads = self.config["model_attrs"]["cv"]["num_heads"]
            self.depth = self.config["model_attrs"]["cv"]["depth"]
            self.latent_dims = eval(
                self.config["model_attrs"]["cv"]["latent_dims"])
            self.resample_dim = self.config["model_attrs"]["cv"]["resample_dim"]
        elif (self.mode == "cls_trans"):
            self.num_layers = self.config["model_attrs"]["cv"]["num_layers"]
            self.dropout_rate = self.config["model_attrs"]["cv"]["dropout_rate"]
            self.patch_size = self.config["model_attrs"]["cv"]["patch_size"]
            self.num_heads = self.config["model_attrs"]["cv"]["num_heads"]
            self.depth = self.config["model_attrs"]["cv"]["depth"]
            self.latent_dims = eval(
                self.config["model_attrs"]["cv"]["latent_dims"])
            self.resample_dim = self.config["model_attrs"]["cv"]["resample_dim"]
        elif (self.mode == "text"):
            self.num_layers = self.config["model_attrs"]["lm"]["num_layers"]
            self.dropout_rate = self.config["model_attrs"]["lm"]["dropout_rate"]
            self.d_model = self.config["model_attrs"]["lm"]["d_model"]
            self.patch_size = self.config["model_attrs"]["lm"]["patch_size"]
            self.num_heads = self.config["model_attrs"]["lm"]["num_heads"]
            self.depth = self.config["model_attrs"]["lm"]["depth"]
            self.latent_dims = eval(
                self.config["model_attrs"]["lm"]["latent_dims"])
            self.resample_dim = self.config["model_attrs"]["lm"]["resample_dim"]
        else:
            raise Exception("Mode not supported, please review config file")

        # Load only the required model configuration
        self.model_attrs = self.config["model_attrs"]["cv"] \
            if self.mode != "text" \
            else self.config["model_attrs"]["lm"]
        if (self.model_attrs["arch"] == "deqformer"):
            from models.architectures.deqformer import Transformer
            self.model = Transformer
        elif (self.model_attrs["arch"] == "mdeqformer"):
            from models.architectures.mdeqformer import Transformer
            self.model = Transformer
        # elif (self.model_attrs["arch"] == "swin"):
        #     from models.architectures.swin import Transformer
        #     self.model = Transformer
        # elif (self.model_attrs["arch"] == "cait"):
        #     from models.architectures.cait import Transformer
        #     self.model = Transformer
        # elif (self.model_attrs["arch"] == "segformer"):
        #     from models.architectures.segformer import Transformer
        #     self.model = Transformer

    def forward_fn(self, data) -> jnp.ndarray:
        """Forward pass."""

        if (self.mode == 'text'):
            from models import transformer_lm
            tokens = data['obs']
            input_mask = jnp.greater(tokens, 0)
            seq_length = tokens.shape[1]
            vocab_size = tokens.shape[-1] * 2  # TODO-clean

            # Embed the input tokens and positions.
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            token_embedding_map = hk.Embed(
                vocab_size, self.d_model, w_init=embed_init)
            token_embs = token_embedding_map(tokens)
            positional_embeddings = hk.get_parameter(
                'pos_embs', [seq_length, self.d_model], init=embed_init)
            x = token_embs + positional_embeddings
            # Run the transformer over the inputs.
            # Transform the transformer
            transformer = transformer_lm.Transformer(
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate
            )
            transformer_pure = hk.transform(transformer)
            z_star = run(self.deq_flag == "True", self.solver, self.mode, x,
                         transformer_pure, input_mask, max_iter=10)

            return hk.Linear(vocab_size)(z_star)

            # TODO: Fix state_with_list updater - non-functional because
            # updater needs to be passed downstream...
        elif (self.mode == 'cls'):
            x = data['obs'].astype('float32')
            num_classes = self.num_classes

            def resnet_fn(x, is_training):
                model = resnet.ResNet18(
                    num_classes=num_classes, resnet_v2=True)
                return model(x, is_training=is_training)

            transformer_cv = hk.transform_with_state(resnet_fn)
            z_star = run(self.config, x,
                         transformer_cv)
            return z_star
        elif (self.mode == 'cls_trans'):

            x = data['obs']

            def cls_fn(x):
                model = self.model(x.shape,
                                   self.patch_size,
                                   self.num_heads,
                                   self.num_classes,
                                   self.depth,
                                   self.resample_dim,
                                   self.mode,
                                   self.latent_dims,
                                   self.config)
                return model(x)

            transformer_cls = hk.transform(cls_fn)
            z_star = run(self.config, x, transformer_cls)

            return z_star

        # TODO: Add fusion modeule or the like...
        elif (self.mode == 'seg'):

            x = data['obs']

            def seg_fn(x):
                model = self.model(x.shape,
                                   self.patch_size,
                                   self.num_heads,
                                   self.num_classes,
                                   self.depth,
                                   self.resample_dim,
                                   self.mode,
                                   self.latent_dims,
                                   self.config)
                return model(x)

            transformer_seg = hk.transform(seg_fn)
            z_star = run(self.config, x, transformer_seg)

            return z_star

    def build_forward_fn(self):
        """Create the model's forward pass."""
        return self.forward_fn
