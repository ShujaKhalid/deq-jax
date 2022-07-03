import jax.numpy as jnp


class Trainor():
    def __init__(self, config):
        super(self, Trainor).__init__():
        self.config = config

    def get_loss_fn(self):
        return loss_fn

    def get_update_fn(self):
        return update_fn

    def get_chk_update_fn(self):
        return chk_update_fn
