import jax
import numpy as np
import jax.numpy as jnp
import tabulate
from sklearn.metrics import classification_report
from tabulate import tabulate


def dice_coeff(logits, y_ohe):
    n = np.bitwise_and(y_ohe, logits)
    # u = np.bitwise_or(y_ohe, logits)
    return 2*np.sum(n)/(y_ohe.sum()+logits.sum())


def jaccard_score(logits, y_ohe):
    n = np.bitwise_and(y_ohe, logits)
    u = np.bitwise_or(y_ohe, logits)
    return np.sum(n)/np.sum(u)


# Jaccard Index
# IoU calculation summed over all classes
def seg_metrics(params, rng, i, x_patch, x, y, ver, save_img_to_folder, forward_fn, config):
    logits = jax.jit(forward_fn.apply)(params, rng, data={
        'obs': x_patch, 'target': y})

    if (config["data_attrs"]["dataset"] == "VOCSegmentation"):
        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    elif (config["data_attrs"]["dataset"] == "Cityscapes"):
        if (config["data_attrs"]["num_classes"] == 20):
            class_names = ['bkgd', 'road', 'sidewalk', 'building', 'wall',
                           'fence', 'pole', 'traffic light', 'traffic sign',
                           'vegetation', 'terrain', 'sky', 'person', 'rider',
                           'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        else:
            class_names = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
                           'dynamic', 'ground', 'road', 'sidewalk',
                           'parking', 'rail track', 'building', 'wall', 'fence',
                           'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
                           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
                           'car', 'truck', 'bus', 'caravan', 'trailer', 'train',
                           'motorcycle', 'bicycle', 'license plate']
    elif (config["data_attrs"]["dataset"] == "MNIST"):
        class_names = ['1', '2', '3', '4', '5',
                       '6', '7', '8', '9']
        # IoU
        # logits_red = jnp.argmax(logits, axis=-1)
        # y_red = jnp.argmax(y, axis=-1)
        # print("y_red: {} - y_hat_red: {}".format(y_red, y_hat_red))

        # binary cross entropy along the different axes
    if (config["data_attrs"]["num_classes"] != 1):
        #print("y.unique(): {}".format(np.unique(y)))

        # dims = y.shape[1] * y.shape[2]
        classes = config["data_attrs"]["num_classes"]
        bs = config["model_attrs"]["batch_size"]
        logits = jax.nn.softmax(logits, axis=-1)
        y_ohe = jax.nn.one_hot(
            y, classes)
        logits_bool = logits.astype(bool)
        y_ohe_bool = y_ohe.astype(bool)
        # print("ver: {} - y: {} - y_hat: {} - y_ohe: {}".format(ver,
        #      y.shape, logits.shape, y_ohe.shape))

        # Jaccard index
        jaccard_classwise = {class_names[c]: np.array([jaccard_score(logits_bool[n, :, :, c], y_ohe_bool[n, :, :, c])
                                                       for n in range(bs)]) for c in range(classes)}
        jaccard_overall = {class_names[i]: v[np.nonzero(v)].mean() for i, v in enumerate(list(
            jaccard_classwise.values())) if np.mean(v) != 0.0}

        # Dice coefficient
        dice_classwise = {class_names[c]: np.array([dice_coeff(logits_bool[n, :, :, c], y_ohe_bool[n, :, :, c])
                          for n in range(bs)])
                          for c in range(classes)}
        dice_overall = {class_names[i]: v[np.nonzero(v)].mean() for i, v in enumerate(list(
            dice_classwise.values())) if np.mean(v) != 0.0}

    # IMPORTANT
    if (i % config["logging"]["save_imgs_step"] == 0):
        save_img_to_folder(config, x, y_ohe, logits, ver)

    return jaccard_overall, dice_overall


# mean average precision
def cls_metrics(params, rng, x, y, forward_fn):
    preds = forward_fn.apply(params, rng, data={'obs': x, 'target': y})

    # Accuracy
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(
        preds, axis=1)
    acc = jnp.mean(predicted_class == target_class)

    rep = classification_report(
        target_class, predicted_class, output_dict=True)
    acc = rep["accuracy"]
    rec = rep["macro avg"]["precision"]
    prec = rep["macro avg"]["recall"]
    f1 = rep["macro avg"]["f1-score"]

    return acc, rec, prec, f1
