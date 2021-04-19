import numpy as np
import warnings
from numpy.core.numeric import NaN
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix

# Metrics to compute

def f1(true: np.ndarray, pred: np.ndarray, num_classes:int) -> dict:
    if true.shape != pred.shape:
        warnings.warn(f"Truth {true.shape} and prediction {pred.shape} shapes should be equal")
        if len(true.shape) == 3:
            true = true[:len(pred),:len(pred[0]),:len(pred[0][0])]
        else:
            true = true[:len(pred),:len(pred[0])]
    labels = np.array(range(num_classes))
    pre = {}
    rec = {}

    for label in labels:
        true_positives = np.sum(np.logical_and(np.equal(true, label), np.equal(pred, label)))
        false_positives = np.sum(np.logical_and(np.logical_not(np.equal(true, label)), np.equal(pred, label)))
        false_negatives = np.sum(np.logical_and(np.equal(true, label), np.logical_not(np.equal(pred, label))))
        rec[f"Rec_{label}"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else NaN 
        pre[f"Pre_{label}"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else NaN
    
    mean_pre = np.nanmean(list(pre.values())) 
    mean_rec = np.nanmean(list(rec.values()))
    f1_dict = {"f1": 2 * mean_pre * mean_rec/(mean_pre + mean_rec), "Pre": mean_pre, "Rec": mean_rec}
    f1_dict.update(pre)
    f1_dict.update(rec)
    return f1_dict

def accuracy(true: np.ndarray, pred: np.ndarray) -> dict:
    if true.shape != pred.shape:
        warnings.warn(f"Truth {true.shape} and prediction {pred.shape} shapes should be equal")
        if len(true.shape) == 3:
            true = true[:len(pred),:len(pred[0]),:len(pred[0][0])]
        else:
            true = true[:len(pred),:len(pred[0])]
    return {"acc": np.mean(np.equal(true,pred))}


def IOU(true: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    if true.shape != pred.shape:
        warnings.warn(f"Truth {true.shape} and prediction {pred.shape} shapes should be equal")
        if len(true.shape) == 3:
            true = true[:len(pred),:len(pred[0]),:len(pred[0][0])]
        else:
            true = true[:len(pred),:len(pred[0])]
    labels = np.array(range(num_classes))
    ious = []
    for label in labels:
        intersection = np.sum(np.logical_and(np.equal(true,label),np.equal(pred,label)))
        union = np.sum(np.logical_or(
            np.equal(true, label), np.equal(pred, label)))
        label_iou = intersection*1.0/union if union > 0 else NaN
        ious.append(label_iou)
    iou_dict = {"IOU_{}".format(label): iou for label,iou in zip(labels,ious)}
    iou_dict["mean_IOU_no_minority"] = np.nanmean(np.delete(ious, 3)) # class 3 is minority class in some datasets
    iou_dict["mean_IOU"] = np.nanmean(ious)
    return iou_dict
def conf_matrix(true: np.ndarray, pred: np.ndarray,num_classes: int) -> dict:
    true = true.flatten()
    pred = pred.flatten()
    conf = confusion_matrix(y_true=true,y_pred=pred,labels=np.arange(num_classes))
    return {"conf_matrix":conf.tolist()}


def evaluation(true: np.ndarray, pred: np.ndarray,num_classes: int) -> dict:
    y_true = np.argmax(true,-1)
    y_pred = np.argmax(pred,-1)
    measures = {}
    measures.update(accuracy(y_true, y_pred))
    measures.update(IOU(y_true, y_pred,num_classes))
    measures.update(f1(y_true, y_pred,num_classes))
    measures.update(hausdorff(y_true,y_pred,num_classes))
    measures.update(conf_matrix(y_true,y_pred,num_classes))
    return measures

def hausdorff(true: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:

    mdict = {}
    haussdorffs = []
    labels = np.array(range(num_classes))
    for clase in labels:
        distanciaClase = []
        TrueM = np.equal(true,clase)
        PredM = np.equal(pred,clase)
        intersec = np.logical_and(TrueM,PredM)
        A = np.logical_and(TrueM,np.logical_not(intersec))
        B = np.logical_and(PredM,np.logical_not(intersec))
        for sample in range(TrueM.shape[0]):
            A_sample = np.where(A[sample])
            B_sample = np.where(B[sample])
            A_sample = [(A_sample[0][i],A_sample[1][i]) for i in range(len(A_sample[0]))]
            B_sample = [(B_sample[0][i],B_sample[1][i]) for i in range(len(B_sample[0]))]

            if len(A_sample) > 0 and len(B_sample) > 0:
                distanciaClase.append(directed_hausdorff(np.array(A_sample),np.array(B_sample)))
        if not len(distanciaClase) == 0:
            mdict[f"Haussdorff_{clase}"] = np.mean(np.array(distanciaClase))
            haussdorffs.append(mdict[f"Haussdorff_{clase}"])
        else:
            mdict[f"Haussdorff_{clase}"] = -1
    mdict[f"mean_Haussdorff"] = np.mean(haussdorffs)
    return mdict

if __name__=="__main__":
    from tensorflow.keras.utils import to_categorical

    true = np.zeros((5,900,1280))
    true[:,:,:1280//2] = 1
    true[:,:1280//2,:] = 3
    true[:,:1280//2,:1280//2] = 2
    true[:,:1280//2,:20] = 4
    pred = np.zeros((5,900,1280))
    pred[:,:,1280//2:] = 2
    pred[:,:1280//2,:] = 4

    print(accuracy(true, pred))
    print(IOU(true, pred))
    print(f1(true, pred))
    print(hausdorff(true,pred))
    
    
