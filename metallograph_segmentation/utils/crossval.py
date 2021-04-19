import numpy as np
import random
import os
import sys
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.backend import clear_session
from . import save, predictions
import json
import torch
RANDOM_SEED=123456789

# Uso:
# train_test(x, y, x_test, {"cropsize": 256}, train_pixelnet, predict_pixelnet)
def train_test(x_train: np.ndarray, y_train: np.ndarray, params: dict, train_f, predict_f, evaluate_f=None, x_test=None,y_test=None, x_val=None, y_val=None, validation_ratio=None, seed=RANDOM_SEED, name="%08x" % random.getrandbits(32)) -> dict:
    debug = params.get("debug", False)
    
    params["name"] = name
    if validation_ratio is not None:
        if debug: print("= Splitting validation subset =", file=sys.stderr)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_ratio, random_state=seed)


    if debug: print("= Training model =", file=sys.stderr)
    model = train_f(params, x_train, y_train, x_val, y_val)
    
    metrics = None
    predictions = None
    if evaluate_f is not None:
        if debug: print("= Computing test metrics =", file=sys.stderr)
        metrics = evaluate_f(model, x_test, y_test, params)
    if x_test is not None:
        if debug: print("= Computing test predictions =", file=sys.stderr)
        predictions = predict_f(model, x_test, params)
        # pre_dict["train"] = predict_f(model, x_train, params)
    del model
    torch.cuda.empty_cache()
    # return model, metrics, predictions
    return metrics, predictions

def saveFolds(folds,foldFile):
    foldDict = {}
    for i, (train_index, val_index) in enumerate(folds):
        foldDict["fold"+str(i)] = {"train":train_index.tolist(),"test":val_index.tolist()}
    
    with open(foldFile,"w") as openFile:
        json.dump(foldDict, openFile)

def loadFolds(foldFile):
    folds = []
    with open(foldFile, "r") as openFile:
        newsFolds = json.load(openFile)

    for key in newsFolds:
        folds.append((np.array(newsFolds[key]["train"]),np.array(newsFolds[key]["test"])))

    return folds
# Uso:
# crossval(x, y, {"cropsize": 256}, train_pixelnet, predict_pixelnet)
def crossval(x: np.ndarray, y: np.ndarray, params: dict, train_f, predict_f, evaluate_f=None, k=5, folds=None, seed=RANDOM_SEED, name="%08x" % random.getrandbits(32)) -> np.ndarray:
    colors = params.get("colors")
    if folds is None:
        folds = list(KFold(k, shuffle=True, random_state=RANDOM_SEED).split(x))
        # saveFolds(folds,"./folds.json")
    
    if not os.path.exists(name):
        os.mkdir(name)

    metrics = []
    pretrained = params.get("pretrained","")

    for i, (train_index, test_index) in enumerate(folds):
        partial_name = f"{name}/fold{i}"
        if not len(pretrained) == 0:
            partial_pretrained = f"{pretrained}/fold{i}"
            params["pretrained"] = partial_pretrained

        if not os.path.exists(partial_name):
            os.mkdir(partial_name)
        clear_session()
        # _, m, fold_predictions = train_test(
        m, fold_predictions = train_test(
            x_train = x[train_index], 
            y_train = y[train_index], 
            params = params, 
            train_f = train_f, 
            predict_f = predict_f, 
            evaluate_f = evaluate_f,
            x_test = x[test_index], 
            y_test = y[test_index],
            validation_ratio = 0.1,
            seed = seed, name = partial_name)
        predictions.save_images(fold_predictions, f"{partial_name}/{os.path.basename(partial_name)}_fold_#.png",
            colors=colors)
        save.save_experiment(params, {"test":m}, name =partial_name, save_dir=params.get("modeldir", params.get("save_dir", "experiments")))
        metrics.append(m)
    
    mdict = {}

    for mname in metrics[0].keys():
        print(mname)
        print([fold[mname] for fold in metrics])
        mdict[mname] = np.nanmean([fold[mname] for fold in metrics]) if None not in [fold[mname] for fold in metrics] else None
        mdict[mname+"_std"] = np.nanstd([fold[mname] for fold in metrics]) if None not in [fold[mname] for fold in metrics] else None

    return mdict
