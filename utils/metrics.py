import jax
import numpy as np
import jax.numpy as jnp


# Jaccard Index
# IoU calculation summed over all classes
def jaccard(params, rng, x_patch, x, y, ver, save_img_to_folder, forward_fn, config):
    logits = jax.jit(forward_fn.apply)(params, rng, data={
        'obs': x_patch, 'target': y})

    # IoU
    # logits_red = jnp.argmax(logits, axis=-1)
    # y_red = jnp.argmax(y, axis=-1)
    # print("y_red: {} - y_hat_red: {}".format(y_red, y_hat_red))

    # binary cross entropy along the different axes
    if (config["data_attrs"]["num_classes"] != 1):
        print("y.unique(): {}".format(np.unique(y)))

        dims = y.shape[1] * y.shape[2]
        logits = jax.nn.softmax(logits, axis=-1)
        y_ohe = jax.nn.one_hot(y, config["data_attrs"]["num_classes"])
        print("ver: {} - y: {} - y_hat: {} - y_ohe: {}".format(ver,
              y.shape, logits.shape, y_ohe.shape))

        y_rsp = y_ohe.reshape(-1, y_ohe.shape[-1]).astype(bool)
        logits_rsp = logits.reshape(-1, logits.shape[-1]).astype(bool)
        print("ver: {} - y_rsp: {} - logits_rsp: {}".format(ver,
              y_rsp.shape, logits_rsp.shape))

        # prep
        # y_rsp = y_rsp != 0.0
        # logits_rsp = logits_rsp != 0.0
        n = np.bitwise_and(y_rsp, logits_rsp)
        u = np.bitwise_or(y_rsp, logits_rsp)
        jaccard_matrix = (n/u).astype(bool)
        # jaccard_matrix[jaccard_matrix != jaccard_matrix] = 0.0
        class_img_map = {cls: np.sum([1.0 if cls in y[b, :, :] else 0.0 for b in range(y.shape[0])]) for cls in range(
            config["data_attrs"]["num_classes"])}
        print("class_exist: {}".format(class_img_map))
        print("class_exist: {}".format(class_img_map[0]))
        print("class_img_map: {}".format(type(class_img_map[0])))
        print("class_img_map: {}".format(np.sum(class_img_map[0])))
        print("jaccard_matrix: {}".format(np.sum(jaccard_matrix[:, 0])))
        # print("sum_check: {}".format(jnp.sum(logits[0, 0, 0, :])))

        # Jaccard class-wise
        jaccard_classwise = {v: np.sum(jaccard_matrix[:, v])/(class_img_map[v]*dims) if class_img_map[v] != 0.0 else 0.0
                             for v in range(jaccard_matrix.shape[-1])}
        print("ver: {} - jaccard_classwise: {}".format(ver, jaccard_classwise))
        jaccard_overall = np.mean(
            [v for v in list(jaccard_classwise.values()) if v != 0.0])
        print("ver: {} - jaccard_overall: {}".format(ver, jaccard_overall))

        # Dice Coefficient
        dice_classwise = {v: 2*np.sum(n[:, v])/(y_rsp[:, v].sum()+logits_rsp[:, v].sum())
                          for v in range(jaccard_matrix.shape[-1])}
        print("ver: {} - dice_classwise: {}".format(ver, dice_classwise))
        dice_overall = np.mean(
            [v for v in list(dice_classwise.values()) if v != 0.0])
        print("ver: {} - dice_overall: {}".format(ver, dice_overall))

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
    # TP = logits_red == y_red

    # print("np.unique(y_hat): {}".format(np.unique(y_hat)))
    # print("np.unique(y): {}".format(np.unique(y)))
    save_img_to_folder(config, x, y_ohe, logits, ver)

    return jaccard_overall


# mean average precision
def accuracy(params, rng, x, y, forward_fn):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(
        forward_fn.apply(params, rng, data={'obs': x, 'target': y}), axis=1)
    return jnp.mean(predicted_class == target_class)
