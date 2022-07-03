import jax
import numpy as np
import jax.numpy as jnp


# Jaccard Index
# IoU calculation summed over all classes
def jaccard(params, rng, x_patch, x, y, save_img_to_folder, forward_fn, config):
    y_hat = jax.jit(forward_fn.apply)(params, rng, data={
        'obs': x_patch, 'target': y})

    # IoU
    y_hat_red = jnp.argmax(y_hat, axis=-1)
    y_red = jnp.argmax(y, axis=-1)

    # print("y_red: {} - y_hat_red: {}".format(y_red, y_hat_red))
    print("y: {} - y_hat: {}".format(y.shape, y_hat.shape))
    y = y
    y_hat = y_hat
    n = np.logical_and(y, y_hat)
    u = np.logical_or(y, y_hat)
    jaccard_matrix = (n/u).astype(bool)
    jaccard_matrix[jaccard_matrix != jaccard_matrix] = 0.0

    # Jaccard class-wise
    jaccard_classwise = {v: np.sum(jaccard_matrix[:, :, :, v])/(y.shape[0] * y.shape[1] * y.shape[2])
                         for v in range(jaccard_matrix.shape[-1])}
    print("jaccard_classwise: {}".format(jaccard_classwise))
    jaccard_overall = np.mean(list(jaccard_classwise.values()))
    print("jaccard_overall: {}".format(jaccard_overall))
    # Dice
    # dice = 2 * (TP) / ( (TP+FP)+(TP+FN) )

    #print("np.unique(y_hat): {}".format(np.unique(y_hat)))
    #print("np.unique(y): {}".format(np.unique(y)))
    #save_img_to_folder(config, x, y, y_hat)

    return jaccard_overall


# mean average precision
def accuracy(params, rng, x, y, forward_fn):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(
        forward_fn.apply(params, rng, data={'obs': x, 'target': y}), axis=1)
    return jnp.mean(predicted_class == target_class)
