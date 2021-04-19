"""Train a USSS model using a config file.

This module helps set up a USSS model for a train-test or cross-validation
execution, extracting evaluation metrics and saving the results in the
specified folder.

    Usage examples:
    
    python -m models.usss.train models/usss/config/default.config
"""

import os
import datetime
import numpy as np
from metallograph_segmentation.utils import data_loader, crossval_multi, metrics, predictions, config, save
from metallograph_segmentation.models.usss.segment_array import train_usss, predict_usss, evaluate_usss
import sys
import argparse
import torch
import pandas as pd
RANDOM_SEED = 123456789
from numpy.random import seed
seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

## Este parámetro limita el número de cpus que utiliza este proceso.
torch.set_num_threads(3)

config_file = "models/usss/config/default.config" if len(sys.argv) <= 1 else sys.argv[1]
conf = config.get_config(config_file)
print("==> CURRENT CONFIGURATION:\n", str(conf), file=sys.stderr)

modeldir = conf.get("modeldir", conf.get("save_dir", "experiments"))
os.makedirs(modeldir, exist_ok=True)

def split_instances(total_instances, first_ratio, *args, **kwargs):
    # reorder the indices randomly and divide onto two parts
    image_distribution = np.arange(total_instances)
    rng = np.random.default_rng()
    rng.shuffle(image_distribution)

    # set the last index of labeled instances
    index_end = np.int32(np.round(first_ratio * total_instances))
    if kwargs.get("log") is not None:
        with open(kwargs.get("log"), "w") as logfile:
            logfile.write(str(image_distribution[:index_end]))
    return [np.array(arg[image_distribution[:index_end]]) for arg in args], [np.array(arg[image_distribution[index_end:]]) for arg in args]


def sample_instances(total_instances, retain_ratio, *args, **kwargs):
    return split_instances(total_instances, retain_ratio, *args, **kwargs)[0]


folds = None
if conf.get("debug"): print("==> START LOADING ===", file=sys.stderr)

default_dataset = {"x": np.array([]), "y": np.array([]), "unlabeled": np.array([]), "x_val": np.array([]), "y_val": np.array([]), "num_labels": 4}
datasets = dict()
folds = dict()

name = os.path.join(modeldir, conf.get("name",
    f"usss_{conf['model']}_{conf['datasets']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))
os.makedirs(name, exist_ok=True)

if "uhcs" in conf["datasets"]:
    datasets["uhcs"] = default_dataset.copy()
    datasets["uhcs"]["x"], datasets["uhcs"]["y"] = data_loader.load_dataset(
        dataset='uhcs',
        path=conf.get("datadir", "data/"),
        image_size=(int(645/16)*16,int(484/16)*16)
    )
    if conf.get("unlabeled-perc") is None and conf.get("labeled-perc") is None:
        # only keep fixed folds if the dataset is not modified by removing labels
        folds_file = "data/uhcs_split.csv"
        folds_idx = pd.read_csv(folds_file)
        folds_idx = folds_idx.values
        folds["uhcs"] = []
        for setIdx in np.unique(folds_idx[:,1]):
            testIdx = np.where(np.equal(folds_idx[:,1],setIdx ) )
            trainIdx = np.where(np.logical_not(np.equal(folds_idx[:,1],setIdx )))
            folds["uhcs"].append((trainIdx,testIdx))
    else:
        total_images = datasets["uhcs"]["x"].shape[0]
        keep_labeled = conf.get("labeled-perc", 100) / 100.
        (datasets["uhcs"]["x"], datasets["uhcs"]["y"]), (remaining_x, _) = split_instances(total_images, keep_labeled, datasets["uhcs"]["x"], datasets["uhcs"]["y"], log=os.path.join(name, "labeled_sample"))
        if conf.get("labeled-perc", 100) < 100:
            keep_unlabeled = conf.get("unlabeled-perc", 100) / (100 - conf.get("labeled-perc", 100))
            datasets["uhcs"]["unlabeled"], = sample_instances(remaining_x.shape[0], keep_unlabeled, remaining_x, log=os.path.join(name, "unlabeled_sample"))
        
        # ---------------------------------- split labeled/unlabeled

    conf["run_test"] = False
    if conf.get("debug"):
        print(x_train.shape, y_train.shape)
        print(np.min(y_train), np.max(y_train))
    
    
if "metaldam" in conf["datasets"]:
    datasets["metaldam"] = default_dataset.copy()
    datasets["metaldam"]["num_labels"] = 5
    datasets["metaldam"]["x"], datasets["metaldam"]["y"] = data_loader.load_dataset(
        dataset="metaldam",
        path=conf.get("datadir", "data/MetalDAM"),
        num_classes = datasets["metaldam"]["num_labels"]#conf.get("num_classes",4)
    )

    # optional: experiment with the amount of labeled/unlabeled instances
    labeled_images = datasets["metaldam"]["x"].shape[0]
    # get the desired percentage of instances
    retain_percentage = conf.get("labeled-perc", 100) / 100.
    datasets["metaldam"]["x"], datasets["metaldam"]["y"] = sample_instances(labeled_images, retain_percentage, datasets["metaldam"]["x"], datasets["metaldam"]["y"], log=os.path.join(name, "labeled_sample"))

    datasets["metaldam"]["unlabeled"], _ = data_loader.load_dataset(
        dataset="additive",
        path=conf.get("unlabeleddir", "data/Additive_unlabeled")
    )
    unlabeled_images = datasets["metaldam"]["unlabeled"].shape[0]

    # get the desired percentage of instances
    retain_percentage = min(conf.get("unlabeled-perc", 100) / 100. * labeled_images / unlabeled_images, 1.)
    datasets["metaldam"]["unlabeled"], = sample_instances(unlabeled_images, retain_percentage, datasets["metaldam"]["unlabeled"], log=os.path.join(name, "unlabeled_sample"))


    conf["run_test"] = False
    if conf.get("debug"):
        print(x_train.shape, y_train.shape)
        print(np.min(y_train), np.max(y_train))


if conf.get("debug"): print("=== END LOADING <==", file=sys.stderr)

if conf.get("dataset","metaldam") == "metaldam":
    colors = [[128, 0, 255], [43, 255, 0], [255, 0, 0], [255, 0, 255],[255, 255, 0]]
else:
    colors = [[255, 255, 0],[0, 0, 255],[21, 180, 214],[75, 179, 36]]

eval_dict = {}
conf["colors"] = colors


if conf.get("run_cv", False):
    eval_val = crossval_multi.crossval(datasets, dict(
        conf), train_usss, predict_usss, evaluate_usss, name=name, k=conf.get("kfold",6),folds=folds)
    eval_dict.update(validation=eval_val)

# with open(f"{name}/{os.path.basename(name)}.out", "w+") as out:
#     out.write(str(eval_val))

if conf.get("run_test", False):
    if conf.get("debug"): print("==> START TRAINING ===", file=sys.stderr)
    (model, (loss, val_loss)), eval_final, test_predictions = crossval_multi.train_test(datasets, dict(conf), train_usss, 
        predict_usss, evaluate_usss, validation_ratio=conf.get("validation_ratio"), name=name)
    if conf.get("debug"): print("=== END TRAINING <==", file=sys.stderr)
    # eval_final = {dname: metrics.evaluation(datasets[dname]["y_test"], test_predictions[dname]["y_test"]) for dname in datasets}
    # eval_final.update(loss=loss, val_loss=val_loss)
    # eval_tr = metrics.evaluation(y_train, test_predictions["train"])
    # eval_dict.update(train=eval_tr, test=eval_final)
    eval_dict.update(test=eval_final)
    for dname in test_predictions:
        predictions.save_images(
            test_predictions[dname]["y_test"], f"{name}/{os.path.basename(name)}_{dname}_test_#.png",colors =colors)

save.save_experiment(
    conf, eval_dict, os.path.basename(name), save_dir=modeldir)
# predictions.save_images(validation_predictions, f"{name}/{os.path.basename(name)}_#.png")
if conf.get("debug"): print("=== END SCRIPT <==", file=sys.stderr)
