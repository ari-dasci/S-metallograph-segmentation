import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import ImageCropper

def normal_density(point, mean, sd):
    constante = 1./(2*np.pi*np.sqrt(np.linalg.det(sd)))

    residuo = np.subtract(point, mean)
    sd_1 = np.linalg.inv(sd)

    scalar = np.matmul(np.matmul(np.transpose(residuo), sd_1), residuo)
    pointwise = np.exp(-0.5 * scalar)

    valor = constante*pointwise
    return valor

def normal_mask(shape):
    weight_mask = np.zeros(shape)
    media = np.array([shape[0]/2, shape[1]/2])
    sd = np.array([[2 * shape[0], 0], [0, 2*shape[1]]])
    for i in range(len(weight_mask)):
        for j in range(len(weight_mask[0])):
            valor = normal_density(np.array([i, j]), media, sd)
            weight_mask[i, j, :] = valor
    return weight_mask
def makeProbabilitiMap(y):
    return np.divide(y,np.repeat(np.sum(y,axis=-1)[:,:,:,np.newaxis],y.shape[-1],axis=-1))

def overlap_predictions(images: np.ndarray, predict_f, target_size=None, nclases=4, step=None, step_fractions=2, weight_mask=None, corners=[0, 1, 2, 3]) -> np.ndarray:
    if target_size is None:
        print(f"Que esta pasando aqui{images.shape}")
        pred = predict_f(images,batch_size=1)
        print(f"Que esta pasando aqui{pred.shape}")
        return pred
    prediction = np.zeros((*(images.shape[0:3]), nclases))

    if weight_mask is None:
        weight_mask = normal_mask((target_size[0], target_size[1], nclases))

    if step is None:
        step = np.int(np.ceil(target_size[0]/step_fractions))

    corner_settings = np.array([
        {"top": True,  "left": True }, # 0: top left
        {"top": True,  "left": False}, # 1: top right
        {"top": False, "left": False}, # 2: bottom right
        {"top": False, "left": True }, # 3: bottom left
    ])[corners]

    for corner_setting in corner_settings:
        cropper = ImageCropper(target_size, step=step, **corner_setting)
        corner_crops = cropper.crop(images)
        # print(f"Crops {corner_crops.shape}")
        corner_prediction = weight_mask * predict_f(corner_crops,batch_size=1)
        # print(f"Predicciones {corner_prediction.shape}")
        corner_prediction = cropper.decrop(corner_prediction)
        prediction[:, cropper.start_y:(cropper.start_y + corner_prediction.shape[1]), 
                   cropper.start_x:(cropper.start_x + corner_prediction.shape[2])] += corner_prediction

    prediction = makeProbabilitiMap(prediction)

    return prediction


def save_images(predictions: np.ndarray, format: str,
    colors =[[255, 255, 0],[0, 0, 255],[21, 180, 214],[75, 179, 36]]) -> list:
    if not format.find("#"):
        ext_loc = format.rfind(".")
        format = format[:ext_loc] + "#" + format[ext_loc:]
    
    name_list = [format.replace("#", str(i)) for i in range(predictions.shape[0])]
    
    colors = np.array(colors)/256.0
    for i in range(predictions.shape[0]):
        predLabels = np.argmax(predictions[i], axis=-1)
        rgb_predLabels = np.zeros(shape=(predLabels.shape[0], predLabels.shape[1],3))

        

        for j in range(4):
            indicesJ = (predLabels == j)
            rgb_predLabels[indicesJ] = colors[j]


        plt.imsave(name_list[i], rgb_predLabels)
    
    return name_list
    

if __name__ == "__main__":
    images = np.array([ [ [1, 2, 3],[3, 4, 5], [5, 6, 7] ], [ [1, 2, 0],[3, 5, 5], [5, 6, 2] ] ])
    labels = np.array([ 
        [ [[1, 2],[2, 4]],[[3,1],[4,2]] ], [ [[1, 2],[2, 4]],[[3,1],[4,2]] ], [ [[1, 2],[2, 4]],[[3,1],[4,2]] ], [ [[1, 2],[2, 4]],[[3,1],[4,2]] ], 
        [ [[1, 2],[3, 4]],[[5,6],[7,8]] ], [ [[1, 2],[3, 4]],[[5,6],[7,8]] ], [ [[1, 2],[3, 4]],[[5,6],[7,8]] ], [ [[1, 2],[3, 4]],[[5,6],[7,8]] ], 
    ])

    pp = overlap_predictions(images, lambda x: labels, target_size=(2, 2), nclases=2, weight_mask=1)
    print(pp[:,:,:,0])
