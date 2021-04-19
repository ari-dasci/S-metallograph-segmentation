#!/usr/bin/python
from metallograph_segmentation.models.stacking import model as px
import os
import datetime
import torch
import pandas as pd
import numpy as np
from metallograph_segmentation.utils import data_loader, crossval, metrics, predictions, config, save
import sys
RANDOM_SEED = 123456789
from numpy.random import seed
seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

## Este parámetro limita el número de cpus que utiliza este proceso.

torch.set_num_threads(3)


config_file = "stacking.config" if len(sys.argv) <= 1 else sys.argv[1]
conf = config.get_config(config_file)
print("==> CURRENT CONFIGURATION:\n", str(conf), file=sys.stderr)

modeldir = conf.get("modeldir", conf.get("save_dir", "experiments"))
os.makedirs(modeldir, exist_ok=True)

folds = None

if conf.get("debug"): print("==> START LOADING ===", file=sys.stderr)
if conf["dataset"] == "uhcs":
    x_train, y_train = data_loader.load_dataset(
        dataset=conf["dataset"],
        path=conf.get("datadir", "../data"),
        image_size=(int(645/16)*16,int(484/16)*16)
    )
    folds_file = "data/uhcs_split.csv"
    folds_idx = pd.read_csv(folds_file)
    folds_idx = folds_idx.values
    folds = []
    for setIdx in np.unique(folds_idx[:,1]):
        testIdx = np.where(np.equal(folds_idx[:,1],setIdx ) )
        trainIdx = np.where(np.logical_not(np.equal(folds_idx[:,1],setIdx )))
        folds.append((trainIdx,testIdx))

    conf["run_test"] = False
    if conf.get("debug"):
        print(x_train.shape, y_train.shape)
        print(np.min(y_train), np.max(y_train))
    
    
elif conf["dataset"] == "metaldam":
    x_train, y_train = data_loader.load_dataset(
        dataset=conf["dataset"],
        path=conf.get("datadir", "../data"),
        num_classes = conf.get("num_classes",4)
    )
    conf["run_test"] = False
    if conf.get("debug"):
        print(x_train.shape, y_train.shape)
        print(np.min(y_train), np.max(y_train))
else:
    (x_train, y_train), (x_test, y_test) = data_loader.load_partitions(
        dataset=conf["dataset"],
        path=conf.get("datadir", "../data"),
        image_size=(int(1280/16)*16,int(900/16)*16)
    )
    if conf.get("debug"):
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
if conf.get("debug"): print("=== END LOADING <==", file=sys.stderr)

if conf.get("dataset","metaldam") == "metaldam":
    colors = [[128, 0, 255], [43, 255, 0], [255, 0, 0], [255, 0, 255],[255, 255, 0]]
else:
    colors = [[255, 255, 0],[0, 0, 255],[21, 180, 214],[75, 179, 36]]
    
name = os.path.join(modeldir, conf.get("name",
    f"stacking_{conf['dataset']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))

if "ST" in name:
    name += "K"+str(conf.get("kernel_size"))
    modelos = conf.get("models")
    for modelo in modelos:
        name+=modelo.split("/")[-1].replace("_efficientnetb0","").replace("imageComplete","IC_")
name = name[:259]


conf["colors"] = colors
eval_dict = {}

if conf.get("run_cv", True):
    x_total, y_total = x_train,y_train
    eval_val = crossval.crossval(x_total, y_total, dict(
        conf), px.train_stackingEnsemble, px.predict_stackingEnsemble, px.evaluate_stackingEnsemble, name=name, k=conf.get("kfold"),folds=folds)
    eval_dict.update(validation=eval_val)

# with open(f"{name}/{os.path.basename(name)}.out", "w+") as out:
#     out.write(str(eval_val))

# if conf.get("run_test", False):
#     if conf.get("debug"): print("==> START TRAINING ===", file=sys.stderr)
#
#     (model, (loss, val_loss)), _, test_predictions = crossval.train_test(x_train, y_train, dict(conf), px.train_stackingEnsemble,
#         px.predict_stackingEnsemble, x_test=x_test, validation_ratio=conf.get("validation_ratio"), name=name)
#     if conf.get("debug"): print("=== END TRAINING <==", file=sys.stderr)
#     eval_final = metrics.evaluation(y_test, test_predictions)
#     eval_final.update(loss=loss, val_loss=val_loss)
#     # eval_tr = metrics.evaluation(y_train, test_predictions["train"])
#     # eval_dict.update(train=eval_tr, test=eval_final)
#     eval_dict.update(test=eval_final)
#     predictions.save_images(
#         test_predictions, f"{name}/{os.path.basename(name)}_test_#.png")
#
# if conf.get("run_infer", False):
#     if conf.get("debug"): print("==> START LOADING MODEL ===", file=sys.stderr)
#     model = px.load_stackingEnsemble(conf["weights"], conf["models"])
#     # (model, (loss, val_loss)), _, test_predictions = crossval.train_test(x_train, y_train, dict(conf), px.train_stackingEnsemble,
#         # px.predict_stackingEnsemble, x_test=x_test, validation_ratio=conf.get("validation_ratio"), name=name)
#     if conf.get("debug"): print("=== END LOADING MODEL <==", file=sys.stderr)
#     test_predictions = px.predict_stackingEnsemble((model, None), x_test, dict(conf))
#     eval_final = metrics.evaluation(y_test, test_predictions)
#     eval_final.update(loss=loss, val_loss=val_loss)
#     # eval_tr = metrics.evaluation(y_train, test_predictions["train"])
#     # eval_dict.update(train=eval_tr, test=eval_final)
#     eval_dict.update(test=eval_final)
#     predictions.save_images(
#         test_predictions, f"{name}/{os.path.basename(name)}_test_#.png")

save.save_experiment(
    conf, eval_dict, os.path.basename(name), save_dir=modeldir)
print(eval_dict)
if conf.get("debug"): print("=== END SCRIPT <==", file=sys.stderr)