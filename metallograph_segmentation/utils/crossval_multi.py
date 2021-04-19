import numpy as np
import random
import os
import sys
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.backend import clear_session
from . import save, predictions
import json
RANDOM_SEED=123456789

# Uso:
# train_test(x, y, x_test, {"cropsize": 256}, train_pixelnet, predict_pixelnet)
def train_test(data: dict, params: dict, train_f, predict_f, evaluate_f=None, validation_ratio=None, seed=RANDOM_SEED, name="%08x" % random.getrandbits(32)) -> dict:
    """
    data = {
        "dataset_name": { "x": np.array, "y": np.array, "x_val": np.array, "y_val": np.array, "x_test": np.array, "y_test": np.array }
    }

    """
    debug = params.get("debug", False)
    
    params["name"] = name
    if validation_ratio is not None:
        training_data = { d: dict() for d in data }
        if debug: print("= Splitting validation subset =", file=sys.stderr)
        for d in data:
            x_train, x_val, y_train, y_val = train_test_split(data[d]["x"], data[d]["y"], test_size=validation_ratio, random_state=seed)
            training_data[d]["x"] = x_train
            training_data[d]["y"] = y_train
            training_data[d]["x_val"] = x_val
            training_data[d]["y_val"] = y_val
            training_data[d]["unlabeled"] = data[d].get("unlabeled", np.array([]))
            training_data[d]["num_labels"] = data[d]["num_labels"]
    else:
        training_data = data.copy()
            
    if debug: print("= Training model =", file=sys.stderr)
    model = train_f(params, training_data)
    
    metrics = None
    predictions = None
    if evaluate_f is not None:
        if debug: print("= Computing test metrics =", file=sys.stderr)
        metrics = evaluate_f(model, data, params)
        # metrics = {
        #     d: evaluate_f(model, data[d]["x_test"], data[d]["y_test"], params)
        #     for d in data if "x_test" in data[d] and "y_test" in data[d]
        # }
    # if "x_test" is not None:
        if debug: print("= Computing test predictions =", file=sys.stderr)
        # predictions = {
        #     d: predict_f(model, data[d]["x_test"], params)
        #     for d in data if "x_test" in data[d]
        # }
        predictions = predict_f(model, data, params)
        # pre_dict["train"] = predict_f(model, x_train, params)
        
    return model, metrics, predictions

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
def crossval(data: dict, params: dict, train_f, predict_f, evaluate_f=None, k=5, folds=None, seed=RANDOM_SEED, name="%08x" % random.getrandbits(32)) -> np.ndarray:
    """
    data = {
        "dataset_name": { "x": np.array, "y": np.array }
    }

    """
    colors = params.get("colors")
    for d in data:
        if folds.get(d) is None:
            folds[d] = list(KFold(k, shuffle=True, random_state=RANDOM_SEED).split(data[d]["x"]))
            saveFolds(folds[d], os.path.join(name, f"./folds_{d}.json"))
    
    if not os.path.exists(name):
        os.mkdir(name)

    metrics = []
    pretrained = params.get("pretrained","")

    for i in range(k):
        current_datasets = {
            d: { "x": data[d]["x"][folds[d][i][0]], "y": data[d]["y"][folds[d][i][0]], 
            "x_test": data[d]["x"][folds[d][i][1]], "y_test": data[d]["y"][folds[d][i][1]],
            "unlabeled": data[d].get("unlabeled", np.array([])), "num_labels": data[d]["num_labels"]  }
            for d in data
        }
        partial_name = f"{name}/fold{i}"
        if not len(pretrained) == 0:
            partial_pretrained = f"{pretrained}/fold{i}"
            params["pretrained"] = partial_pretrained

        if not os.path.exists(partial_name):
            os.mkdir(partial_name)
        clear_session()

        _, m, fold_predictions = train_test(
            data = current_datasets,
            params = params, 
            train_f = train_f, 
            predict_f = predict_f, 
            evaluate_f = evaluate_f,
            validation_ratio = 0.1,
            seed = seed, name = partial_name)
        
        for dname in fold_predictions:
            predictions.save_images(fold_predictions[dname]["y_test"], f"{partial_name}/{os.path.basename(partial_name)}_{dname}_fold_#.png",
                colors=colors)
        save.save_experiment(params, {"test":m}, name = partial_name, save_dir=params.get("modeldir", params.get("save_dir", "./experiments")))
        metrics.append(m)
    
    mdict = {dname: {} for dname in metrics[0]} # meter aquí medias por dataset

    for dname in mdict:
    # for mname in metrics[0].keys():
        for mname in metrics[0][dname]:
            mdict[dname][mname] = np.mean([fold[dname][mname] for fold in metrics])
            mdict[dname][mname+"_std"] = np.std([fold[dname][mname] for fold in metrics])

    return mdict
