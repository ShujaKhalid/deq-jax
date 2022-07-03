import jax
import jax.numpy as jnp
from typing import Any, Mapping


class Losses():
    def __init__(self, config, forward_fn):
        self.config = config
        self.forward_fn = forward_fn
        self.mode = self.config["mode"]
        self.num_classes = self.config["data_attrs"]["num_classes"]

    def lm_loss_fn(self,
                   params,
                   rng,
                   data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        """Compute the loss on data wrt params."""
        logits = self.forward_fn(params, rng, data, is_training)
        targets = jax.nn.one_hot(data['target'], self.num_classes)
        print("logits.shape: {} - targets.shape: {}".format(logits.shape, targets.shape))
        assert logits.shape == targets.shape

        mask = jnp.greater(data['obs'], 0)
        loss = jnp.sum(-jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        return loss

    def vm_loss_fn(self,
                   params,
                   rng,
                   data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        """Compute the loss on data wrt params."""
        # print('data.shape: {}'.format(data))
        logits = self.forward_fn(params, rng, data, is_training)
        targets = jax.nn.one_hot(data['target'], self.num_classes)
        print("logits.shape: {} - targets.shape: {}".format(logits.shape, targets.shape))
        assert logits.shape == targets.shape
        loss = jnp.sum(-jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

        return loss

    def seg_loss_fn(self,
                    params,
                    rng,
                    data: Mapping[str, jnp.ndarray]
                    ) -> jnp.ndarray:
        """Compute the loss on data wrt params."""
        # print('data.shape: {}'.format(data))
        logits = self.forward_fn.apply(params, rng, data)
        targets = jax.nn.one_hot(data['target'], self.num_classes)
        print("logits.shape: {} - targets.shape: {}".format(logits.shape, targets.shape))
        assert logits.shape == targets.shape
        loss = jnp.sum(-jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

        return loss

    def get_loss_fn(self):
        if (self.mode == "text"):
            return self.lm_loss_fn
        elif (self.mode == "cls" or self.mode == "cls_trans"):
            return self.vm_loss_fn
        elif (self.mode == "seg"):
            return self.seg_loss_fn
        else:
            raise Exception(
                "Invalid mode selected... Please refer to configuration file")
