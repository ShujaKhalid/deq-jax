import jax
import functools
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
from tabulate import tabulate
from utils.utils import save_img_to_folder


def evaluate_cls(rng, state, epoch, config, ds_dict, preproc, cls_metrics):
    res_trn = {
        "acc": [],
        "rec": [],
        "prec": [],
        "f1": []
    }
    res_tst = {
        "acc": [],
        "rec": [],
        "prec": [],
        "f1": []
    }
    log_policy = eval(config["logging"]["log_policy"])
    if ("train" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_trn'])):
            x = preproc(x, config)
            acc, rec, prec, f1 = cls_metrics(state['params'],
                                             rng,
                                             x,
                                             jax.nn.one_hot(y, config["data_attrs"]["num_classes"]))
            res_trn["acc"].append(acc)
            res_trn["rec"].append(rec)
            res_trn["prec"].append(prec)
            res_trn["f1"].append(f1)
    if ("valid" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_tst'])):
            x = preproc(x, config)
            acc, rec, prec, f1 = cls_metrics(state['params'],
                                             rng,
                                             x,
                                             jax.nn.one_hot(y, config["data_attrs"]["num_classes"]))
            res_tst["acc"].append(acc)
            res_tst["rec"].append(rec)
            res_tst["prec"].append(prec)
            res_tst["f1"].append(f1)

    headers = ["metric", "value"]
    table_trn = [["TRAIN_"+k, np.mean(v)]
                 for k, v in zip(res_trn.keys(), res_trn.values())]
    table_tst = [["TEST_"+k, np.mean(v)]
                 for k, v in zip(res_tst.keys(), res_tst.values())]

    print(tabulate(table_trn, headers, tablefmt="fancy_grid"))
    print(tabulate(table_tst, headers, tablefmt="fancy_grid"))

    # print("  epoch: {} - iter: {} \
    #         - acc_trn {:.2f} - acc_tst: {:.2f} \
    #         - rec_trn {:.2f} - rec_tst: {:.2f} \
    #         - prec_trn {:.2f} - prec_tst: {:.2f} \
    #         - f1_trn {:.2f} - f1_tst: {:.2f} "
    #       .format(epoch,
    #               i,
    #               np.mean(res_trn["acc"]), np.mean(res_tst["acc"]),
    #               np.mean(res_trn["rec"]), np.mean(res_tst["rec"]),
    #               np.mean(res_trn["prec"]), np.mean(res_tst["prec"]),
    #               np.mean(res_trn["f1"]), np.mean(res_tst["f1"])))


def evaluate_seg(rng, state, epoch, config, ds_dict, preproc, seg_metrics):
    jac_trn = []
    jac_val = []
    dice_trn = []
    dice_val = []
    log_policy = eval(config["logging"]["log_policy"])
    trn_set_policy = config["logging"]["trn_set"]
    tst_set_policy = config["logging"]["tst_set"]

    if ("train" in log_policy):
        ver = "train"
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_trn'])):
            # print("x (before preproc): {}".format(x))
            # print("y: {}".format(y))
            x_patch = jnp.array(preproc(x, config))
            # print("np.unique(y): {}".format(np.unique(y)))
            jac, dice = seg_metrics(state['params'],
                                    rng,
                                    i,
                                    x_patch,
                                    x,
                                    y,
                                    ver,
                                    functools.partial(save_img_to_folder, i, epoch))
            jac_trn.append(jac)
            dice_trn.append(dice)

            if (i == trn_set_policy):
                break

    if ("valid" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_tst'])):
            ver = "val"
            # print("x (before preproc): {}".format(x))
            # print("y: {}".format(y))
            x_patch = jnp.array(preproc(x, config))
            # print("np.unique(y): {}".format(np.unique(y)))
            jac, dice = seg_metrics(state['params'],
                                    rng,
                                    i,
                                    x_patch,
                                    x,
                                    y,
                                    ver,
                                    functools.partial(save_img_to_folder, i, epoch))
            jac_val.append(jac)
            dice_val.append(dice)

            if (i == tst_set_policy):
                break

    # Train - Jaccard
    headers = ["Class", "Jaccard Index"]
    classes = jac_trn[0].keys()
    units = len(jac_trn)
    jaccard_trn = [[cls, np.mean([jac_trn[m][cls] for m in range(units) if cls in list(jac_trn[m].keys())])]
                   for _, cls in enumerate(classes)]
    # Test - Jaccard
    classes = jac_val[0].keys()
    units = len(jac_val)
    jaccard_val = [[cls, np.mean([jac_val[m][cls] for m in range(units) if cls in list(jac_val[m].keys())])]
                   for _, cls in enumerate(classes)]

    # Train - Dice
    headers = ["Class", "Dice Co-efficient"]
    classes = dice_trn[0].keys()
    units = len(dice_trn)
    dice_trn = [[cls, np.mean([dice_trn[m][cls] for m in range(units) if cls in list(dice_trn[m].keys())])]
                for _, cls in enumerate(classes)]
    # Test - Dice
    classes = dice_val[0].keys()
    units = len(dice_val)
    dice_val = [[cls, np.mean([dice_val[m][cls] for m in range(units) if cls in list(dice_val[m].keys())])]
                for _, cls in enumerate(classes)]

    print("===> TRAINING <===")
    print(tabulate(jaccard_trn, headers, tablefmt="fancy_grid"))
    print(tabulate(jaccard_val, headers, tablefmt="fancy_grid"))
    print("===> VALIDATION <===")
    print(tabulate(dice_trn, headers, tablefmt="fancy_grid"))
    print(tabulate(dice_val, headers, tablefmt="fancy_grid"))

    # print(tabulate(table_trn, headers, tablefmt="fancy_grid"))
    print("epoch: {} - iter: {} - jac_trn {:.2f} - jac_val: {:.2f} - dice_trn {:.2f} - dice_val: {:.2f}".format(epoch, i,
                                                                                                                np.mean(np.array(jaccard_trn)[
                                                                                                                        :, 1].astype(float)),
                                                                                                                np.mean(np.array(jaccard_val)[
                                                                                                                        :, 1].astype(float)),
                                                                                                                np.mean(np.array(dice_trn)[
                                                                                                                        :, 1].astype(float)),
                                                                                                                np.mean(np.array(dice_val)[
                                                                                                                        :, 1].astype(float)),))
