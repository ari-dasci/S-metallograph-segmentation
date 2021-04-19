#!/usr/bin/python3
import os
import numpy as np
from ...utils.preprocessing import ImageCropper, applyDA, processInput
from ...utils.predictions import overlap_predictions
from ...utils.metrics import evaluation
from ...utils.losses import my_losses
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim import Adam, SGD, AdamW
import torch
from torch.utils.tensorboard import SummaryWriter

from ..pixelnet.model import PixelNet

import segmentation_models_pytorch as sm


def get_sm(params):
    # if len(params.get("pretrained","")) != 0:
    #     return get_pretrained_sm(params)

    model = params.get("model")
    backbone = params.get("backbone")
    num_classes = params.get("num_classes")

    modeloFinal = None
    if model == "fpn":
        modeloFinal =  sm.FPN(encoder_name=backbone,classes = num_classes)
    elif model == "linknet":
        modeloFinal =  sm.Linknet(encoder_name=backbone,classes = num_classes)
    elif model == "pspnet":
        modeloFinal =  sm.PSPNet(encoder_name=backbone,classes = num_classes)
    elif model == "unet":
        modeloFinal =  sm.Unet(encoder_name=backbone,classes = num_classes)
    elif model == "unetplusplus":
        modeloFinal = sm.UnetPlusPlus(encoder_name=backbone,classes = num_classes)
    elif model == "deeplabv3":
        modeloFinal =  sm.DeepLabV3(encoder_name=backbone,classes = num_classes)
    elif model == "deeplabv3plus":
        modeloFinal =  sm.DeepLabV3Plus(encoder_name=backbone,classes = num_classes)
    elif model == "manet":
        modeloFinal =  sm.MAnet(encoder_name=backbone,classes = num_classes)
    elif model == "pan":
        modeloFinal =  sm.PAN(encoder_name=backbone,classes = num_classes)
    elif model == "pixelnet":
        modeloFinal = PixelNet(n_classes = num_classes)
    else:
        print("No model named",model)
    # print(summary(modeloFinal.to("cuda"), (3,224,224)))
    return modeloFinal

'''
def get_pretrained_sm(params:dict):
    model           = params.get("model")
    backbone        = params.get("backbone")
    num_classes     = params.get("num_classes")
    pre_num_clases  = params.get("pre_num_clases")
    pretrained      = params.get("pretrained")
    pretrained      = f"{pretrained}/weights.best.hdf5"
    paramsPretrained = {
        "model":model,
        "backbone":backbone,
        "num_clases":pre_num_clases,
        "weights_file":pretrained
    }
    pre_model = load_sm(paramsPretrained)

    final_Output = torch.nn.Conv2d(in_channels=pre_model.,out_channels=num_classes,kernel_size=3,padding="same")(pre_model.layers[-3].output)
    final_Output = torch.nn.Softmax()(final_Output)

    model = Model(pre_model.input,final_Output)

    return model
'''

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1  # if channels last
        #axis=  1 #if channels first

        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index
        classSelectors = torch.argmax(true, axis=axis)
        #if your loss is sparse, use only true as classSelectors

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        classSelectors = [torch.equal(np.int64(i), classSelectors)
                          for i in range(len(weightsList))]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        classSelectors = [x.type(torch.float32) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(classSelectors, weightsList)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true, pred)
        loss = loss * weightMultiplier

        return loss
    return lossFunc

def LossFunction(params: dict):
    num_classes =       int(params.get("num_classes", 4))
    weighted    =       bool(params.get('weighted_loss',False))
    weights     =       np.array(params.get("weights",np.ones(shape=num_classes)))
    weights     =       torch.Tensor(np.array([float(w) for w in weights]))
    loss        =       params.get("loss","categorical_crossentropy")
    continuity  =       params.get("continuiti",False)

    loss_name = loss
    if 'categorical_crossentropy' in loss:
        loss = my_losses()[loss]()
    else:
        
        loss = my_losses()[loss](mode='multilabel')
        loss.__name__ = loss_name
    # loss = my_losses()[loss](mode='multilabel')

    if params.get("loss","categorical_crossentropy") == "categorical_crossentropy":
        loss.weight = weights
    '''
    if weighted:
        elementos = []
        for elem in np.unique(np.argmax(y, axis=-1)):
            elementos.append(np.sum(np.equal(elem, np.argmax(y, axis=-1))))
        class_weight = [np.max(elementos)/elemento for elemento in elementos]
        class_weight = np.multiply(class_weight,weights)
        loss = weightedLoss(loss,class_weight)
    '''
    if continuity:
        loss = my_losses()["continuiti_loss"](loss)

    return loss

def Optimizer(params: dict,model):
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

def Metrics(params:dict):
    metrics = [
        sm.utils.metrics.IoU(),
        sm.utils.metrics.Fscore(),
        sm.utils.metrics.Accuracy(),
        sm.utils.metrics.Precision(),
        sm.utils.metrics.Recall()
    ]
    return metrics

# Devuelve el modelo entrenado fcnn y la función de pérdida de validación y de train de la mejor época
# respecto a la función de pérdida en validación
def train_sm(params: dict, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, **kwargs):

    writer = SummaryWriter(log_dir=os.path.join(params['name'], 'logs'))
    
    device      =       torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses      =       0,0
    DA          =       bool(params.get("da", True))
    batch_size  =       params.get("batch_size",4)

    model       =       get_sm(params)
    save_dir    =       params.get('name')
    epochs      =       int(params.get("epochs",200))
    save_file   =       os.path.join(save_dir, 'weights.best.pt')
    soft_labelling =    params.get("soft_labelling",False)
    training_with_swa = params.get("training_with_swa",False)


    lossF = LossFunction(params)
    # lossF.__name__ = params.get("loss")
    # lossF = DiceLoss()

    opt = Optimizer(params,model)
    metrics = Metrics(params)

    x,y = processInput(x,y,params,soft_labelling)
    x_val,y_val = processInput(x_val,y_val,params)


    x = torch.Tensor(x.copy())
    y = torch.Tensor(y.copy()).type(torch.long)

    x_val = torch.Tensor(x_val.copy())
    y_val = torch.Tensor(y_val.copy()).type(torch.long)

    if training_with_swa:
        x = torch.cat([x,x_val])
        y = torch.cat([y,y_val])

    trainDS = torch.utils.data.TensorDataset(x,y)

    if DA :
        trainDS = applyDA(trainDS)
        
    valDS = torch.utils.data.TensorDataset(x_val,y_val)

    

    trainLoader = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True, drop_last=True, )
    valLoader = torch.utils.data.DataLoader(valDS,shuffle=False)
    ## Defining how to train an epoch

    train_epoch = sm.utils.train.TrainEpoch(
        model, 
        loss=lossF, 
        metrics=metrics, 
        optimizer=opt,
        device=device,
        verbose=True,
    )
    valid_epoch = sm.utils.train.ValidEpoch(
        model, 
        loss=lossF, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    ## Actually training x epochs

    if training_with_swa == False:
        best_val_loss = None
        best_val_loss_train = None
        for i in range(epochs):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(trainLoader)
            valid_logs = valid_epoch.run(valLoader)

            writer.add_scalar('Loss/train', train_logs[lossF.__name__], i)
            writer.add_scalar('Loss/val', valid_logs[lossF.__name__], i)
            writer.add_scalar('IOU/train', train_logs['iou_score'], i)
            writer.add_scalar('IOU/val', valid_logs['iou_score'], i)
            writer.add_scalar('F1/train', train_logs['fscore'], i)
            writer.add_scalar('F1/val', valid_logs['fscore'], i)
            writer.add_scalar('Accuracy/train', train_logs['accuracy'], i)
            writer.add_scalar('Accuracy/val', valid_logs['accuracy'], i)
            writer.add_scalar('Precision/train', train_logs['precision'], i)
            writer.add_scalar('Precision/val', valid_logs['precision'], i)
            writer.add_scalar('Recall/train', train_logs['recall'], i)
            writer.add_scalar('Recall/val', valid_logs['recall'], i)

            if best_val_loss is None or best_val_loss > valid_logs[lossF.__name__]:
                best_val_loss = valid_logs[lossF.__name__]
                best_val_loss_train = train_logs[lossF.__name__]
                torch.save(model.state_dict(), save_file)
                print(f"New best model reached. Best val_loss:{best_val_loss}")
            # if best_val_loss is None or best_val_loss > valid_logs[params.get("loss")]:
            #     best_val_loss = valid_logs[params.get("loss")]
            #     best_val_loss_train = train_logs[params.get("loss")]
            #     torch.save(model.state_dict(), save_file)
            #     print(f"New best model reached. Best val_loss:{best_val_loss}")
        state_dict = torch.load(save_file)
        model.load_state_dict(state_dict)

        return model, ( best_val_loss_train , best_val_loss )
    else:
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(opt, T_max=100)
        swa_start = 5
        swa_scheduler = SWALR(opt, swa_lr=0.05)

        for i in range(epochs):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(trainLoader)
            torch.save(model.state_dict(), save_file)
            if i > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        swa_model.cpu()
        swa_model.eval()
        torch.optim.swa_utils.update_bn(trainLoader, swa_model)
        
        torch.save(swa_model.state_dict(), save_file)
        
        return swa_model, (train_logs[lossF.__name__],None)

# Acepta el modelo entrenado anteriormente y las imágenes que debemos predecir.

def predict_sm(result, x: np.ndarray, params: dict):
    device      =       torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _    =       result
    num_classes =       params.get("num_classes",4)

    x,_ = processInput(x.copy(),y=None,parameters=params)

    x = torch.Tensor(x)


    with torch.no_grad(): # No realizar cálculos de gradiente innecesarios en inferencia para evitar problemas de memoria.
        # model = model.to(device)
        # x = x.to(device)
        # finalShape = np.array(x.shape)
        # pred = model(x[0:2]).cpu().detach().numpy()
        #
        # finalShape[1] = pred.shape[1]
        # predicciones = np.zeros(shape=finalShape)
        #
        # if not params.get("overlap",False):
        #     for step in range(len(x)//batch_size):
        #         predStep = model(x[step*batch_size:(step+1)*batch_size])
        #         predStep = predStep.to("cpu")
        #         predStep = predStep.detach().numpy()
        #         predicciones[step*batch_size:(step+1)*batch_size] = predStep
        #     predicciones = np.swapaxes(predicciones,1,2)
        #     predicciones = np.swapaxes(predicciones,2,3)
        #     return predicciones
        model = model.to(device)
        x = x.to(device)

        if not params.get("overlap", False):
            predicciones = torch.zeros((x.shape[0],num_classes,x.shape[2],x.shape[3]))
            '''
            for i in range(len(x)):
                predicciones[i:i+1] = model(x[i:i+1])
            '''
            predicciones = model(x)
            predicciones = predicciones.to("cpu")
            predicciones = predicciones.detach().numpy()
            predicciones = np.swapaxes(predicciones, 1, 2)
            predicciones = np.swapaxes(predicciones, 2, 3)

            # Free up memory
            del x
            torch.cuda.empty_cache()

            return predicciones

    # train/predict con cropping no implementado para pytorch.
    cropsize = params.get("cropsize",448)
    sizeImages = (cropsize,cropsize)
    num_classes = int(params.get("num_classes", 4))
    if params.get("reconstruct", True):
        mask = np.ones(shape = (sizeImages[0],sizeImages[1],num_classes))
        return overlap_predictions(images=x, predict_f=model.predict, target_size=sizeImages, nclases=num_classes,step=params.get("step",sizeImages[0]//4),weight_mask = mask)
    else:
        cropper = ImageCropper(sizeImages)
        crops = cropper.crop(x)
        return cropper.decrop(model.predict(crops))
    


def evaluate_sm(model, x: np.ndarray, y: np.ndarray, params: dict):
    _ ,(tr_loss,val_loss) = model
    preds = predict_sm(model, x, params)

    # metrics = Metrics(params)
    # for metric in metrics:
    #     print(metric.__name__,np.array(metric(torch.Tensor(preds),torch.Tensor(y))))

    eval_dict = evaluation(y,preds,num_classes = params["num_classes"])
    eval_dict.update(loss=tr_loss, val_loss=val_loss)
    return eval_dict

def load_sm(params):
    
    weights_file = params.get("weights_file")
    # print(weights_file)
    model = get_sm(params)
    state_dict = torch.load(weights_file)
    if params.get("training_with_swa"):
        model = AveragedModel(model)
    model.load_state_dict(state_dict)
    return model
