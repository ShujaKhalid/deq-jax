import jax.numpy as jnp
import haiku as hk

from typing import Mapping
from models import resnet

from utils.utils import run


class Forward():
    def __init__(self, config, data):
        super(self, Forward).__init__()
        self.config = config
        self.data = data
        self.mode = self.config["mode"]
        self.deq_flag = self.config["deq_params"]["deq_flag"]
        self.max_steps = self.config["deq_params"]["max_steps"]
        self.solver = self.config["deq_params"]["solver"]
        self.num_classes = self.config["data_params"]["num_classes"]
        if (self.mode == "cv"):
            self.num_layers = config["model_params"]["cv"]["num_layers"]
            self.dropout_rate = config["model_params"]["cv"]["dropout_rate"]
            self.batch_size = config["model_params"]["cv"]["batch_size"]
            self.d_model = config["model_params"]["cv"]["d_model"]
            self.patch_size = config["model_params"]["cv"]["patch_size"]
            self.num_heads = config["model_params"]["cv"]["num_heads"]
            self.depth = config["model_params"]["cv"]["depth"]
            self.latent_dims = eval(
                config["model_params"]["cv"]["latent_dims"])
            self.resample_dim = config["model_params"]["cv"]["resample_dim"]
        elif (self.mode == "text"):
            self.num_layers = config["model_params"]["lm"]["num_layers"]
            self.dropout_rate = config["model_params"]["lm"]["dropout_rate"]
            self.batch_size = config["model_params"]["lm"]["batch_size"]
            self.d_model = config["model_params"]["lm"]["d_model"]
            self.patch_size = config["model_params"]["lm"]["patch_size"]
            self.num_heads = config["model_params"]["lm"]["num_heads"]
            self.depth = config["model_params"]["lm"]["depth"]
            self.latent_dims = eval(
                config["model_params"]["lm"]["latent_dims"])
            self.resample_dim = config["model_params"]["lm"]["resample_dim"]
        else:
            raise Exception("Mode not supported, please review config file")

    def forward_fn(self, data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
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
            from models.transformer_cv import TransformerCV
            x = data['obs'].astype('float32')

            def cls_fn(x):
                model = TransformerCV(x.shape,
                                      self.patch_size,
                                      self.num_heads,
                                      self.num_classes,
                                      self.depth,
                                      self.resample_dim,
                                      self.latent_dims)
                return model(x)

            transformer_cls = hk.transform(cls_fn)
            z_star = run(self.config, x, transformer_cls)

            return z_star

        # TODO: Add fusion modeule or the like...
        elif (self.mode == 'seg'):
            from models.transformer_cv import TransformerCV

            x = data['obs'].astype('float32')

            def seg_fn(x):
                model = TransformerCV(x.shape,
                                      self.patch_size,
                                      self.num_heads,
                                      self.num_classes,
                                      self.depth,
                                      self.resample_dim,
                                      self.latent_dims)
                return model(x)

            transformer_seg = hk.transform(seg_fn)
            z_star = run(self.config, x, transformer_seg)

            return z_star

    def build_forward_fn(self):
        """Create the model's forward pass."""
        return self.forward_fn()
