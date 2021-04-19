#!/usr/bin/python3
import os
from collections import namedtuple
import json
import sys
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ...utils.preprocessing import gray_to_3Channel, gray_repeat3, cropImages, ImageCropper, applyDA, processInput
import random
from ...utils.predictions import overlap_predictions
from ...utils.metrics import evaluation
from ...utils.losses import my_losses
import datetime
from ..sm import model as sm
import segmentation_models_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
import torchvision

class StackingEnsemble(nn.Module):

    def __init__(self, n_features, n_labels, kernel_size, dropout_rate, padding):
        super(StackingEnsemble, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_features,
                              out_channels=n_labels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

def Optimizer(params: dict, model):
    opt             = params.get("optimizer")
    learning_rate   = params.get("learning_rate")
    optimizer = None
    if opt == "adam":
        optimizer = Adam(model.parameters(),lr=learning_rate,betas=(float(params.get("beta1", 0.9)),float(params.get("beta2", 0.999))))
    elif opt == "sgd":
        optimizer = SGD(model.parameters(),lr=learning_rate)
    elif opt == "adamw":
        optimizer = AdamW(model.parameters(),lr=learning_rate,betas=(float(params.get("beta1", 0.9)),float(params.get("beta2", 0.999))))

    return optimizer

def Metrics():
    metrics = [
        segmentation_models_pytorch.utils.metrics.IoU(),
        segmentation_models_pytorch.utils.metrics.Fscore(),
        segmentation_models_pytorch.utils.metrics.Accuracy(),
        segmentation_models_pytorch.utils.metrics.Precision(),
        segmentation_models_pytorch.utils.metrics.Recall()
    ]
    return metrics

def LossFunction(params: dict):
    loss        =       params.get("loss", "categorical_crossentropy")
    loss = my_losses()[loss]()
    return loss

def get_modeloST(modelo,backbone,nclasses):
    aux_params = {
        "weights_file": modelo,
        "backbone": backbone,
        "num_classes": nclasses
    }

    if "deeplabv3plus" in modelo:
        aux_params["model"] = "deeplabv3plus"
    elif "deeplabv3" in modelo:
        aux_params["model"] = "deeplabv3"
    elif "fpn" in modelo:
        aux_params["model"] = "fpn"
    elif "linknet" in modelo:
        aux_params["model"] = "linknet"
    elif "pan" in modelo:
        aux_params["model"] = "pan"
    elif "pspnet" in modelo:
        aux_params["model"] = "pspnet"
    elif "unetplusplus" in modelo:
        aux_params["model"] = "unetplusplus"
    elif "unet" in modelo:
        aux_params["model"] = "unet"
    elif "pixelnet" in modelo:
        aux_params["model"] = "pixelnet"

    model = sm.load_sm(aux_params)
    return model

def concatenateModelsPredictions(x,models,backbones):
    # inps will be a stack of predictions
    inps = None
    for modelName,backbone in zip(models,backbones):
        if "fold" in modelName:
            paremetersModelFile = modelName[:-len("fold0/weights.best.pt")]+"parameters.json"
            # foldI = modelName[-len("fold0/weights.best.pt"):-len("/weights.best.pt")]
        else:
            paremetersModelFile = modelName[:-len("weights.best.pt")]+"parameters.json"
        
        with open(paremetersModelFile,"r") as file:
            parametersModel = json.load(file)

        model = get_modeloST(modelName, backbone, parametersModel.get("num_classes"))
        y = sm.predict_sm((model,None), x, parametersModel)

        # Concatenate predictions
        if inps is None:
            inps = y
        else:
            inps = np.concatenate((inps, y),axis=-1)
    return inps

def train_stackingEnsemble(params: dict, x: np.ndarray, y: np.ndarray, x_val=None, y_val=None, **kwargs):
    num_classes = int(params.get("num_classes", 4))
    epochs      =       int(params.get("epochs",200))
    batch_size  =       int(params.get("batch_size", 4))
    save_dir    =       params.get('name')
    kernel_size =       params.get("kernel_size",1)
    modelos     =       params.get("models",None)
    backbones   =       params.get("backbones",None)
    dropout_rate =       float(params.get("dropoutrate",0.2))
    save_file = os.path.join(save_dir, 'weights.best.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if modelos == None:
        print("No model to stack")
        exit(-1)

    postfix = save_dir.split("/")[-1]+"/weights.best.pt"
    if "fold" in postfix:
        modelos = [modelo+"/"+postfix for modelo in modelos]
    else:
        modelos = [modelo+"/weights.best.pt" for modelo in modelos]

    x = concatenateModelsPredictions(x, modelos, backbones)
    x = np.swapaxes(x, 2, 3)
    x = np.swapaxes(x, 1, 2)
    y = np.swapaxes(y, 2, 3)
    y = np.swapaxes(y, 1, 2)

    x_val = concatenateModelsPredictions(x_val, modelos, backbones)
    x_val = np.swapaxes(x_val, 2, 3)
    x_val = np.swapaxes(x_val, 1, 2)
    y_val = np.swapaxes(y_val, 2, 3)
    y_val = np.swapaxes(y_val, 1, 2)

    padding = (kernel_size-1)//2

    model = StackingEnsemble(n_features=x.shape[1], n_labels=num_classes, kernel_size=kernel_size,
                             dropout_rate=dropout_rate, padding=padding)

    opt = Optimizer(params, model)
    lossF = LossFunction(params)
    metrics = Metrics()

    model.to(device)

    for metric in metrics:
        metric.to(device)

    x = torch.Tensor(x.copy()).to(device)
    y = torch.Tensor(y.copy()).type(torch.long).to(device)
    x_val = torch.Tensor(x_val.copy()).to(device)
    y_val = torch.Tensor(y_val.copy()).type(torch.long).to(device)

    trainDS = torch.utils.data.TensorDataset(x, y)
    valDS = torch.utils.data.TensorDataset(x_val, y_val)

    trainLoader = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valDS, shuffle=False)

    train_epoch = segmentation_models_pytorch.utils.train.TrainEpoch(
        model,
        loss=lossF,
        metrics=metrics,
        optimizer=opt,
        device=device,
        verbose=True,
    )

    valid_epoch = segmentation_models_pytorch.utils.train.ValidEpoch(
        model,
        loss=lossF,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    best_val_loss = None
    best_val_loss_train = None
    for i in range(epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(trainLoader)
        valid_logs = valid_epoch.run(valLoader)

        if best_val_loss is None or best_val_loss > valid_logs["categorical_cross_entropy_loss"]:
            best_val_loss = valid_logs["categorical_cross_entropy_loss"]
            best_val_loss_train = train_logs["categorical_cross_entropy_loss"]
            torch.save(model.state_dict(), save_file)
            print(f"New best model reached. Best val_loss:{best_val_loss}")

    state_dict = torch.load(save_file)
    model.load_state_dict(state_dict)

    return model, (best_val_loss_train, best_val_loss)

def predict_stackingEnsemble(result, x: np.ndarray, params: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = result

    modelos = params.get("models", None)
    save_dir = params.get("name")
    backbones   = params.get("backbones",None)
    postfix = save_dir.split("/")[-1]+"/weights.best.pt"
    if "fold" in postfix:
        modelos = [modelo+"/"+postfix for modelo in modelos]
    else:
        modelos = [modelo+"/weights.best.pt" for modelo in modelos]

    x = concatenateModelsPredictions(x, modelos,backbones)
    x = np.swapaxes(x, 2, 3)
    x = np.swapaxes(x, 1, 2)
    x = torch.Tensor(x.copy()).to(device)

    with torch.no_grad():  # Do not perform unnecessary gradient calculations in inference to avoid memory problems
        model = model.to(device)
        predicciones = model.predict(x)
        predicciones = predicciones.to("cpu")
        predicciones = predicciones.detach().numpy()
        predicciones = np.swapaxes(predicciones, 1, 2)
        predicciones = np.swapaxes(predicciones, 2, 3)

        return predicciones

def evaluate_stackingEnsemble(model, x: np.ndarray, y: np.ndarray, params: dict):
    _, (tr_loss,val_loss) = model
    preds = predict_stackingEnsemble(model, x, params)
    eval_dict = evaluation(y,preds,num_classes=params["num_classes"])
    eval_dict.update(loss=tr_loss, val_loss=val_loss)
    return eval_dict


# def load_stackingEnsemble(weights_file, modelosBase, nclasses=4, **kwargs):
#     model = get_stackingEnsemble(len(modelosBase), nclasses)
#     model.load_weights(weights_file)
#     return model
