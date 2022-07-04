import jax
import numpy as np
import jax.numpy as jnp


# Jaccard Index
# IoU calculation summed over all classes
def jaccard(params, rng, x_patch, x, y, save_img_to_folder, forward_fn, config):
    logits = jax.jit(forward_fn.apply)(params, rng, data={
        'obs': x_patch, 'target': y})

    # IoU
    # logits_red = jnp.argmax(logits, axis=-1)
    # y_red = jnp.argmax(y, axis=-1)
    # print("y_red: {} - y_hat_red: {}".format(y_red, y_hat_red))

    # binary cross entropy along the different axes
    if (config["data_attrs"]["num_classes"] != 1):
        print("y.unique(): {}".format(np.unique(y)))

        logits = jax.nn.softmax(logits, axis=-1)
        y = jax.nn.one_hot(y, config["data_attrs"]["num_classes"])
        print("y: {} - y_hat: {}".format(y.shape, logits.shape))

        y_rsp = y.reshape(-1, y.shape[-1]).astype(bool)
        logits_rsp = logits.reshape(-1, logits.shape[-1]).astype(bool)
        print("y_rsp: {} - logits_rsp: {}".format(y_rsp.shape, logits_rsp.shape))

        # prep
        n = np.bitwise_and(y_rsp, logits_rsp)
        u = np.bitwise_or(y_rsp, logits_rsp)
        jaccard_matrix = (n/u).astype(bool)
        jaccard_matrix[jaccard_matrix != jaccard_matrix] = 0.0
        # print("sum_check: {}".format(jnp.sum(logits[0, 0, 0, :])))

        # Jaccard class-wise
        jaccard_classwise = {v: np.sum(jaccard_matrix[:, v])/(y_rsp.shape[0])
                             for v in range(jaccard_matrix.shape[-1])}
        print("jaccard_classwise: {}".format(jaccard_classwise))
        jaccard_overall = np.mean(list(jaccard_classwise.values()))
        print("jaccard_overall: {}".format(jaccard_overall))

        # Dice Coefficient
        dice_classwise = {v: 2*np.sum(n[:, v])/(y_rsp[:, v].sum()+logits_rsp[:, v].sum())
                          for v in range(jaccard_matrix.shape[-1])}
        print("dice_classwise: {}".format(dice_classwise))
        dice_overall = np.mean(list(dice_classwise.values()))
        print("dice_overall: {}".format(dice_overall))

    else:
        print("y: {} - y_hat: {}".format(y.shape, logits.shape))
        logits = np.squeeze(logits.astype(int))
        n = np.logical_and(y, logits)
        u = np.logical_or(y, logits)
        print("y.unique(): {}".format(np.unique(y)))
        print("y_hat.unique(): {}".format(np.unique(logits)))
        jaccard_matrix = (n/u).astype(bool)
        jaccard_matrix[jaccard_matrix != jaccard_matrix] = 0.0
        jaccard_overall = np.sum(
            jaccard_matrix[:, :, :])/(y.shape[0] * y.shape[1] * y.shape[2])
        print("jaccard_overall: {}".format(jaccard_overall))
    # Dice
    # dice = 2 * (TP) / ( (TP+FP)+(TP+FN) )
    #TP = logits_red == y_red

    #print("np.unique(y_hat): {}".format(np.unique(y_hat)))
    #print("np.unique(y): {}".format(np.unique(y)))
    save_img_to_folder(config, x, y, logits)

    return jaccard_overall


# mean average precision
def accuracy(params, rng, x, y, forward_fn):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(
        forward_fn.apply(params, rng, data={'obs': x, 'target': y}), axis=1)
    return jnp.mean(predicted_class == target_class)
