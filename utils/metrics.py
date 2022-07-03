import jax.numpy as jnp


# Jaccard similarity
def jaccard(params, rng, x_patch, x, y, save_img_to_folder, forward_fn):
    y_hat = jax.jit(forward_fn.apply)(params, rng, data={
        'obs': x_patch, 'target': y}, is_training=False)
    # print("x.shape: {}".format(x.shape))
    # print("y.shape: {}".format(y.shape))
    # print("y_hat.shape: {}".format(y_hat.shape))
    print("np.unique(y_hat): {}".format(np.unique(y_hat)))
    print("np.unique(y): {}".format(np.unique(y)))
    save_img_to_folder(config, x, y, y_hat)
    xt = y_hat != 0
    yt = y != 0
    num = jnp.sum(jnp.logical_xor(
        xt, yt).astype(jnp.int32))
    denom = jnp.sum(jnp.logical_or(jnp.logical_and(
        xt, yt), jnp.logical_xor(xt, yt)).astype(jnp.int32))
    return jnp.where(denom == 0, 0.0, num.astype(jnp.float32) / denom)


# mean average precision
def accuracy(params, rng, x, y, forward_fn):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(
        forward_fn.apply(params, rng, data={'obs': x, 'target': y}, is_training=False), axis=1)
    return jnp.mean(predicted_class == target_class)
