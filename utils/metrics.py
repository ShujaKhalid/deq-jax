import jax
import numpy as np
import jax.numpy as jnp
from sklearn.metrics import jaccard_score


def dice_coeff(y_ohe, logits, v):
    n = np.bitwise_and(y_ohe, logits)
    # u = np.bitwise_or(y_ohe, logits)
    return 2*np.sum(n[:, :, :, v])/(y_ohe[:, :, :, v].sum()+logits[:, :, :, v].sum())


# Jaccard Index
# IoU calculation summed over all classes
def jaccard(params, rng, x_patch, x, y, ver, save_img_to_folder, forward_fn, config):
    logits = jax.jit(forward_fn.apply)(params, rng, data={
        'obs': x_patch, 'target': y})

    if (config["data_attrs"]["dataset"] == "VOCSegmentation"):
        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    elif (config["data_attrs"]["dataset"] == "Cityscapes"):
        class_names = []
        # IoU
        # logits_red = jnp.argmax(logits, axis=-1)
        # y_red = jnp.argmax(y, axis=-1)
        # print("y_red: {} - y_hat_red: {}".format(y_red, y_hat_red))

        # binary cross entropy along the different axes
    if (config["data_attrs"]["num_classes"] != 1):
        print("y.unique(): {}".format(np.unique(y)))

        # dims = y.shape[1] * y.shape[2]
        classes = config["data_attrs"]["num_classes"]
        logits = jax.nn.softmax(logits, axis=-1).astype(bool)
        y_ohe = jax.nn.one_hot(
            y, classes).astype(bool)
        print("ver: {} - y: {} - y_hat: {} - y_ohe: {}".format(ver,
              y.shape, logits.shape, y_ohe.shape))

        # Jaccard index
        jaccard_classwise = {class_names[c]: np.array([jaccard_score(logits[n, :, :, c].flatten(), y_ohe[n, :, :, c].flatten())
                                                       for n in range(y_ohe[..., c].shape[0])]) for c in range(classes)}
        # print(jaccard_classwise)
        jaccard_overall = [v[np.nonzero(v)].mean() for v in list(
            jaccard_classwise.values()) if np.mean(v) != 0.0]
        print("ver: {} - jaccard_overall: {}".format(ver, jaccard_overall))
        jaccard_overall = np.mean(jaccard_overall)

        # Dice coefficient
        dice_classwise = {class_names[c]: np.array([dice_coeff(y_ohe, logits, c)
                                                    for n in range(y_ohe[..., c].shape[0])]) for c in range(classes)}
        # print(jaccard_classwise)
        dice_overall = [v[np.nonzero(v)].mean() for v in list(
            dice_classwise.values()) if np.mean(v) != 0.0]
        print("ver: {} - dice_overall: {}".format(ver, dice_overall))
        dice_overall = np.mean(dice_overall)

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

    return jaccard_overall, dice_overall


# mean average precision
def accuracy(params, rng, x, y, forward_fn):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(
        forward_fn.apply(params, rng, data={'obs': x, 'target': y}), axis=1)
    return jnp.mean(predicted_class == target_class)
