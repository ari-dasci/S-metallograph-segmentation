"""Original (modified) training script for USSS

This module implements the training process for a PyTorch-based Universal
semi-supervised segmentation model. It has been modified to accept a dict
of arguments and a testing subset.
"""

import os
import time
import numpy as np
import torch, sys

from PIL import Image, ImageOps
from argparse import ArgumentParser
from .EntropyLoss import EmbeddingLoss
from .iouEval import iouEval#, getColorEntry

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import torchvision
import torch.nn.functional as F


# from .dataset_loader import *
# from . import transform as transforms
from ...utils import metrics, save
from ...utils.preprocessing import gray_to_3Channel, gray_repeat3, cropImages, ImageCropper, applyDA, processInput

import importlib
from collections import OrderedDict , namedtuple

from shutil import copyfile
import matplotlib as mp
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.utils import to_categorical
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR

def empty_data_loader():
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(np.array([]))))

def train_usss(params: dict, datasets: dict, **kwargs):
    """
    Structure of datasets:

    {
        "dataset_name": {
            "x": np.array, 
            "y": np.array,
            "unlabeled": np.array,
            "x_val": np.array, 
            "y_val": np.array
        }
    }

    """

    # Read parameters
    epochs = int(params.get("epochs", params.get("num-epochs", 100)))
    lr = float(params.get("lr", 1e-3))
    num_workers = int(params.get("num-workers", 0))
    batch_size = int(params.get("batch-size", 6))
    model_type = params.get("model", "models.usss.drnet")
    finetune = bool(params.get("finetune", False))
    steps_loss = int(params.get("steps-loss", 50))
    pacc = bool(params.get("pacc", True))
    epochs_save = int(params.get("epochs-save", 0))
    augment_data = bool(params.get("da", True))
    swa = bool(params.get("swa", True))
    best_acc = 0

    # if swa:
    #     return train_usss_swa(params, datasets, **kwargs)

    # Configure log paths
    savedir = params.get("name", ".")
    loss_logpath = savedir + "/loss_log.txt"
    enc = kwargs.get("enc", False)

    if enc:
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"  

    # GPU check
    n_gpus = torch.cuda.device_count()
    print("\nWorking with {} GPUs".format(n_gpus))

    # Check datasets
    names = list(datasets.keys())

    alpha = int(params.get("alpha", len(names) > 1))
    beta = int(params.get("beta", len(names) > 1))

    entropy = (alpha + beta) > 0

    if alpha > 0:
        assert len(names) > 1 , "Inter-dataset entropy Module undefined with single dataset. Exiting ... "

    print("Working with {} Dataset(s):".format(len(names)))
    for key, d in datasets.items():
        # print(datasets[key])
        print("{}[{}]: Unlabeled images {}, Training on {} images, Validation on {} images".format(key, d["num_labels"], len(d["unlabeled"]), len(d["x"]) , len(d["x_val"])))


    # Create model objects
    num_labels = {dname: d["num_labels"] for dname, d in datasets.items()}
    em_dim=int(params.get("em_dim", 100))
    model = create_model(
        model_module=model_type,
        bnsync=bool(params.get("bnsync", False)),
        num_labels=num_labels,
        em_dim=em_dim,
        resnet=params.get("resnet", "resnet_18"),
        state=params.get("state", None)
    )

    # Start log files
    if not os.path.exists(automated_log_path):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            if len(datasets) > 1:
                myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU-1\t\tTrain-IoU-2\t\tVal-IoU-1\t\tVal-IoU-2\t\tlearningRate")
            else:
                myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tVal-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    if not os.path.exists(loss_logpath):
        with open(loss_logpath , "w") as myfile:
            if len(datasets) > 1:
                myfile.write("Epoch\t\tS1\t\tS2\t\tUS1\t\tUS2\t\tTotal\n")
            else:
                myfile.write("Epoch\t\tS1\t\tS2\t\tTotal\n")

    # Set up optimizer and cuda
    if 'drnet' in model_type:
        optimizer = SGD(model.optim_parameters(), lr, 0.9, weight_decay=1e-4)     ## scheduler DR-Net
    if bool(params.get("cuda", True)):
        model = torch.nn.DataParallel(model).cuda()
    if swa:
        swa_model = AveragedModel(model)

    doIou = {
        'train': bool(params.get("ioutrain", True)),
        'val': bool(params.get("iouval", True))
    }
    le_file = savedir + '/label_embedding.pt'
    average_epoch_loss = { 'train': np.inf, 'val': np.inf }
    best_epoch_loss = { 'train': np.inf, 'val': np.inf }

    label_embedding = {key: torch.randn(num_labels[key], em_dim).cuda() for key in names} ## Random Initialization

    ## If provided, use label embedddings
    pt_em = params.get("pt_em")
    if pt_em:
        fn = torch.load(pt_em)
        label_embedding = {key: torch.tensor(fn[key], dtype=torch.float).cuda() for key in names}

    # Possibly resume from checkpoint
    start_epoch = 1
    if bool(params.get("resume", False)):
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        if os.path.exists(filenameCheckpoint):
            checkpoint = torch.load(filenameCheckpoint)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if swa:
                swa_state_dict = checkpoint['swa_state_dict']
                if swa_state_dict is not None:
                    swa_model.load_state_dict(swa_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['best_acc']
            label_embedding = torch.load(le_file) #if len(datasets) >1 else None
            print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))
        else:
            print("WARNING: resume option was used but checkpoint was not found in folder. Training from scratch!")

    swa_start = int(np.floor(.6 * epochs))
    if swa:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        swa_scheduler = SWALR(optimizer, swa_lr=lr)
    else:
        swa_model = model
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1-((epoch-1)/epochs)),0.9))  ## scheduler 2
    # loss_criterion = {key:torch.nn.CrossEntropyLoss(ignore_index=NUM_LABELS[key]-1).cuda() for key in datasets}
    # class weights may be added here:
    loss_criterion = {key:torch.nn.CrossEntropyLoss().cuda() for key in datasets}

    # Prepare datasets and create data loader objects
    loader_train = dict()
    loader_val = dict()
    loader_unlabeled = dict()

    for d in names:
        x, y = processInput(datasets[d]["x"], datasets[d]["y"], params)

        x = torch.Tensor(x.copy())
        y = torch.Tensor(y.copy()).type(torch.long)

        
        tsds = TensorDataset(x, y)
        if augment_data:
            tsds = applyDA(tsds)
        loader_train[d] = DataLoader(tsds, num_workers=num_workers, batch_size=batch_size, 
                            shuffle=True)

        if datasets[d]["x_val"] is not None:
            x_val, y_val = processInput(datasets[d]["x_val"], datasets[d]["y_val"], params)
            x_val = torch.Tensor(x_val.copy())
            y_val = torch.Tensor(y_val.copy()).type(torch.long)
            tsds = TensorDataset(x_val, y_val)
            loader_val[d] = DataLoader(tsds, num_workers=num_workers, batch_size=1, 
                                shuffle=False, drop_last=True)
        elif not swa:
            print("No validation subset provided! Set the validation_ratio option in the config")
            exit(1)

        if entropy:
            if "unlabeled" in datasets[d] and len(datasets[d]["unlabeled"]) > 0:
                unlabeled, _ = processInput(datasets[d]["unlabeled"], None, params)
                
                unlabeled = torch.Tensor(unlabeled.copy())
                tsds = TensorDataset(unlabeled)
                if augment_data:
                    tsds = applyDA(tsds, labeled=False)
                loader_unlabeled[d] = DataLoader(unlabeled, num_workers=num_workers, batch_size=batch_size, 
                                    shuffle=True, drop_last=True)
            else:
                loader_unlabeled[d] = empty_data_loader()

    # Define embedding losses if needed
    if entropy:
        similarity_module = EmbeddingLoss(num_labels, em_dim, label_embedding, loss_criterion)
        similarity_module = torch.nn.DataParallel(similarity_module).cuda()
    torch.save(label_embedding, le_file)

    
    print()
    print("========== STARTING TRAINING ===========")
    print()

    n_iters = min([len(loader_train[d]) for d in names])

    if entropy:
        unlabeled_iters = {d:len(loader_unlabeled[d])//n_iters for d in names}

    
    for epoch in (range(start_epoch, epochs+1) if epochs >= start_epoch else [-1]):

        epoch_start_time = time.time()
        usedLr = 0
        iou = {key: (0,0) for key in names}
    
        ###### TRAIN begins  #################
        if epoch > 0:
            for phase in ['train']:

                eval_iou = doIou[phase]
                print("-----", phase ,"- EPOCH", epoch, "-----")
  
                model.train()

                for param_group in optimizer.param_groups:
                    print("LEARNING RATE: " , param_group['lr'])
                    usedLr = float(param_group['lr'])

                ## Initialize the iterables
                # print("1")
                labeled_iterator = {dname:iter(loader_train[dname]) for dname in names}
                # print("2")
                if entropy:
                    unlabeled_iterator = {dname:iter(loader_unlabeled[dname]) for dname in names}

                # if alpha:
                #     alpha = 1
                # if beta:
                #     beta = 1

                epoch_loss = {d:[] for d in names}
                epoch_sup_loss = {d:[] for d in names}
                epoch_ent_loss = {d:[] for d in names}

                time_taken = []    
                # print("3")
                if eval_iou:
                    iou_data = {key:iouEval(num_labels[key], -1) for key in names}

                for itr in range(n_iters):
                    # print("4[{}]".format(itr))
                    optimizer.zero_grad()
                    loss_sup = {d:0 for d in names}
                    loss_ent = {d:[0] for d in names}

                    for d in names:
                        # print("[ Generating instances from {} ]".format(d))
                        images_l , targets_l = next(labeled_iterator[d])
                        images_l = images_l.cuda()
                        targets_l_nm = targets_l.argmax(1).cuda()

                        start_time = time.time()

                        # print("[ Forward ]")
                        dec_outputs = model(images_l , enc=False, finetune=finetune)
                        # print(dec_outputs[d].shape)

                        ## DEBUG: content of predictions
                        if bool(params.get("debug", False)):
                            t = targets_l.data.cpu().numpy()
                            p = torch.nn.LogSoftmax().cuda()(dec_outputs[d]).data.cpu().numpy()
                            print(np.shape(p))
                            print(f"=debug= {d} [{np.mean(t)}+-{np.std(t)}]: {np.mean(p)} +- {np.std(p)}")
                        ###

                        loss_s = loss_criterion[d](dec_outputs[d], targets_l_nm)
                        loss_s.backward()

                        if eval_iou:
                            iou_data[d].addBatch(dec_outputs[d].argmax(1, True).data, targets_l)

                        loss_sup[d] = loss_s.item()

                        if entropy:
                            # print("[ unsupervised loss ]")
                            for _ in range(unlabeled_iters[d]):

                                images_u = next(unlabeled_iterator[d])
                                images_u = images_u.cuda()

                                _ , en_outputs = model(images_u)

                                loss_e = torch.mean(similarity_module(en_outputs, d, alpha, beta)) ## unsupervised losses
                                loss_e /= unlabeled_iters[d]
                                loss_e.backward()
                                loss_ent[d].append(loss_e.item())

                        epoch_sup_loss[d].append(loss_sup[d])
                        epoch_ent_loss[d].extend(loss_ent[d])
                        epoch_loss[d].append(loss_sup[d] + np.sum(loss_ent[d])) ## Already averaged over iters

                    time_taken.append(time.time() - start_time)
                    optimizer.step()


                    if steps_loss > 0 and (itr % steps_loss == 0 or itr == n_iters-1):
                        average = {d:np.around(sum(epoch_loss[d]) / len(epoch_loss[d]) , 3) for d in names}
                        print(f'{phase} loss: {average} (epoch: {epoch}, step: {itr})', 
                                "// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / batch_size))
                
                    best_epoch_loss[phase] = np.sum([np.min(epoch_loss[d]) for d in names])
                    average = {d:np.mean(epoch_loss[d]) for d in names}	
                    average_epoch_loss[phase] = sum(average.values())

                    if entropy:
                        average_epoch_sup_loss = {d:np.mean(epoch_sup_loss[d]) for d in names}
                        average_epoch_ent_loss = {d:np.mean(epoch_ent_loss[d]) for d in names}

                        ## Write the epoch wise supervised and total unsupervised losses. 
                        with open(loss_logpath , "a") as myfile:
                            if len(datasets) > 1 and (itr % steps_loss == 0 or itr == n_iters-1):
                                myfile.write("%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n"%(
                                    epoch, average_epoch_sup_loss.get(names[0], 0), average_epoch_sup_loss.get(names[1], 0),
                                    average_epoch_ent_loss.get(names[0], 0), average_epoch_ent_loss.get(names[1], 0),
                                    average_epoch_loss[phase]))
                            

                ## Todo: A better way to close the worker threads.
                for d in names:
                    while True:
                        try:
                            _ =  next(labeled_iterator[d])
                        except StopIteration:
                            break;

                    if entropy:
                        while True: 
                            try:
                                _ =  next(unlabeled_iterator[d])
                            except StopIteration:
                                break;

                iou = {key:(0,0) for key in names}

                if (eval_iou):
                    iou = {key:iou_data[key].getIoU() for key in names}

                    iouStr_label = {key : '{:0.2f}'.format(iou[key][0]*100) for key in names}
                    for d in names:
                        print ("EPOCH IoU on {} dataset: {} %".format(d , iouStr_label[d]))

                if swa and epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()
            
            ########## Train ends ###############################

        ##### Validation ###############
        ## validation after every 5 epoch,
        ## save after validation or save last model if swa is being used
        validation_step = not swa and ((epoch == 1) or (epoch % 5 == 0))
        saving_step = swa or validation_step

        if validation_step:
            for phase in ['val']:
                eval_iou = doIou[phase]
                print("-----", phase ,"- EPOCH", epoch, "-----")

                model.eval()
                
                if eval_iou:
                    iou_data = {d:iouEval(num_labels[d], -1) for d in names}

                epoch_val_loss = {d:[] for d in names}
                if pacc:
                    pAcc = {d:[] for d in names}

                for d in names:
                    time_taken = []    

                    for itr, (images, targets) in enumerate(loader_val[d]):
                        start_time = time.time()

                        images = images.cuda()
                        targets = targets.argmax(1, True).cuda()

                        with torch.set_grad_enabled(False):                            
                            seg_output = model(images, enc=False)
                            loss = loss_criterion[d](seg_output[d], targets.squeeze(1))
    
                            # print("step2: evaliou")
                            if eval_iou:
                                pred = seg_output[d]
                                # print(f"pred {pred.shape}")
                                pred = pred.argmax(1, True).data
                                # print(f"pred.argmax {pred.shape}")
                                # print(f"targets {targets.data.shape}")
                                iou_data[d].addBatch(pred, targets.data)
                                if pacc:
                                    a = (pred == targets.data)
                                    pAcc[d].append(torch.mean(a.double()))

                            epoch_val_loss[d].append(loss.item())

                        time_taken.append(time.time() - start_time)

                        if steps_loss > 0 and (itr % steps_loss == 0 or itr == len(loader_val[d])-1):
                            average = np.around(np.mean(epoch_val_loss[d]) , 3)
                            print(f'{d}: {phase} loss: {average} (epoch: {epoch}, step: {itr})', 
                                    "// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / batch_size)) 
                            
                best_epoch_loss[phase] = np.sum([np.min(epoch_val_loss[d]) for d in names])
                average_epoch_loss[phase] = np.sum([np.mean(epoch_val_loss[d]) for d in names])

                if (eval_iou):
                    iou = {d:iou_data[d].getIoU() for d in names}

                    iouStr_label = {d : '{:0.2f}'.format(iou[d][0]*100) for d in names}
                    for d in names:
                        print ("EPOCH IoU on {} dataset: {} %".format(d , iouStr_label[d]))
                        if pacc:
                            print(f'{d}: pAcc : {np.mean(pAcc[d])*100}%')
            ############# VALIDATION ends #######################

        print("Epoch time {} s".format(time.time() - epoch_start_time))
        if saving_step:

            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
            # filename = f'{savedir}/model-{epoch:03}.pth'
            # filenamebest = f'{savedir}/model_best.pth'

            if epoch > 0:
                # remember best valIoU and save checkpoint
                if sum([iou[key][0] for key in names]) == 0:
                    current_acc = -average_epoch_loss['val']
                else:
                    current_acc = sum([iou[key][0] for key in names])/len(names) ## Average of the IoUs to save best model

                is_best = current_acc > best_acc
                best_acc = max(current_acc, best_acc)
                
                save_dict = {
                    'epoch': epoch + 1,
                    'arch': str(model),
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }
                if swa:
                    save_dict.update(swa_state_dict=swa_model.state_dict())
                save_checkpoint(save_dict, is_best or swa, filenameCheckpoint, filenameBest)

                if is_best and not swa:
                    with open(savedir + "/best.txt", "w") as myfile:
                        myfile.write("Best epoch is %d\n" % (epoch))
                        for d in names:
                            myfile.write("Val-IoU-%s= %.4f\n" % (d, iou[d][0]))

                        myfile.write("\n\n")   
                        
                        for d in names:
                            myfile.write("Classwise IoU for best epoch in %s is ... \n" % (d))
                            for values in iou[d][1]:
                                myfile.write("%.4f "%(values))
                            myfile.write("\n\n")        
            ### END saving

        with open(automated_log_path, "a") as myfile:
            iouTrain = 0
            if swa:
                if len(names) > 1:
                    myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['train'], iouTrain, iouTrain, iou[list(names)[0]][0], iou[list(names)[1]][0], usedLr ))
                else:
                    myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['train'], iouTrain, iou[list(names)[0]][0], usedLr ))
            else:
                if len(names) > 1:
                    myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], iouTrain, iouTrain, iou[list(names)[0]][0], iou[list(names)[1]][0], usedLr ))
                else:
                    myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], iouTrain, iou[list(names)[0]][0], usedLr ))
        # if epoch > 0
    # for epoch


    if swa:
        for d in loader_train:
            torch.optim.swa_utils.update_bn(loader_train[d], swa_model)
    else:
        model.load_state_dict(torch.load(filenameBest)["state_dict"])

    swa_model.eval()
    return swa_model, (best_epoch_loss["train"], 0 if swa else best_epoch_loss["val"])

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    """Save a model checkpoint

    Args:
        state: model state to be saved
        is_best: boolean indicating if model is current best
        filenameCheckpoint: normal checkpoint file name
        filenameBest: file where model will be saved if it is the best
    """
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def create_model(model_module, bnsync, num_labels, em_dim, resnet, state):
    model_file = importlib.import_module("." + model_module, package="metallograph_segmentation.models.usss")
    if bnsync:
        model_file.BatchNorm = batchnormsync.BatchNormSync
    else:
        model_file.BatchNorm = torch.nn.BatchNorm2d

    model = model_file.Net(num_labels, em_dim, resnet)
    
    if state:
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
            for name, param in state_dict.items():
                
                if name.startswith(('seg' , 'up' , 'en_map' , 'en_up')):
                    continue
                elif name not in own_state:
                    print("Not loading {}".format(name))
                    continue
                own_state[name].copy_(param)

            print("Loaded pretrained model ... ")
            return model

        
        state_dict = torch.load(state)
        model = load_my_state_dict(model, state_dict)
    return model

def predict_usss(result, datasets: dict, params: dict=dict()):
    """
    Datasets dict looks like: 
    {
        "dataset_name": { "x": np.array/Dataset/torch.Tensor }
    }
    """
    model, _ = result

    print("=== PREDICTING TEST ===")

    dataset_predictions = {d: { "y": None } for d in datasets}

    for dlabel in datasets:
            print(f"Now predicting {dlabel}:")
        # try:
            # print(datasets[dlabel]["x_test"][0])
            x, y = processInput(datasets[dlabel]["x_test"], None, params)
            thedata = torch.Tensor(x.copy())
            thedata = TensorDataset(thedata)
            loader = DataLoader(thedata, num_workers=int(params.get("num_workers", 0)), batch_size=1, 
                shuffle=False, drop_last=False)
            img_iterator = iter(loader)
            
            total_lab, total_pred = None, None

            for (images,) in img_iterator:
                # images, labels = next(img_iterator)
                images = images.cuda()

                with torch.set_grad_enabled(False):
                    seg_output = model(images, enc=False)
                
                predictions = seg_output[dlabel].data.cpu().numpy()
                # print(predictions)

                # need to expand the label matrix into binary form,
                # discard the channel dimension
                # num_labels is hardcoded here, should parameterize it
                # labels = to_categorical(labels[:,0,...], 5 if dlabel == 'metaldam' else 4)
                # channels (classes) are axis 1 in Torch, transpose for evaluation
                predictions = np.transpose(predictions, [0, 2, 3, 1])
                if total_pred is None:
                    # total_lab = labels
                    total_pred = predictions
                else:
                    # total_lab = np.concatenate((total_lab, labels), 0)
                    total_pred = np.concatenate((total_pred, predictions), 0)
            
            dataset_predictions[dlabel]["y_test"] = total_pred
        # except Exception:
            # print(f"Error while testing dataset {dlabel}. Please set the resume flag and try again. Exception:")

    return dataset_predictions

def evaluate_usss(result, datasets: dict, params: dict):
    _ ,(tr_loss, val_loss) = result
    preds = predict_usss(result, datasets, params)

    # metrics = Metrics(params)
    # for metric in metrics:
    #     print(metric.__name__,np.array(metric(torch.Tensor(preds),torch.Tensor(y))))

    dataset_evaluations = { d: None for d in datasets }
    for d in datasets:
        dataset_evaluations[d] = metrics.evaluation(datasets[d]["y_test"], preds[d]["y_test"], datasets[d].get("num_labels", 4))
        dataset_evaluations[d].update(loss=tr_loss, val_loss=val_loss)

    return dataset_evaluations
            