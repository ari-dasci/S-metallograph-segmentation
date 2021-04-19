from os import truncate
import numpy as np
import warnings
from scipy import ndimage
from skimage.feature import local_binary_pattern as LBP
import cv2
import PIL
import numbers
import torch
import torchvision
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

class ImageCropper(object):
    """Helper class for cropping training and test images

    Examples:
        >>> train_cropper = ImageCropper((256, 256))
        >>> crops_train, labels_train = train_cropper.crop(x_train, y_train)
        >>> test_cropper = ImageCropper((256, 256))
        >>> crops_test, labels_test = test_cropper.crop(x_test, y_test)
        >>> crops_pred = train_test(crops_train, crops_test, {}, train_pixelnet, predict_pixelnet)
        >>> predictions = test_cropper.decrop(crops_pred)
        >>> save_predictions(predictions)
    """
    def __init__(self, crop_shape=(224, 224), step=None, save_name=None, top=True, left=True):
        """Initializer

        Init method for ImageCropper, saves the parameters.

        Args:
            crop_shape (tuple): desired height and width of each crop
            step: length that is skipped from a crop to the next, by default a whole crop size will be skipped
            save_name: unused
            top: start cropping from the top border?
            left: start cropping from the left border?
        """
        self.crop_shape = crop_shape
        self.save_name = save_name
        self.step = step
        self.cropped = False
        self.top = top
        self.left = left

    def crop(self, images, labels=None):
        """Cropping method

        Receives a set of images and, optionally, labels, and returns cropped versions of them.

        Args:
            images: Array of input images
            labels: Array of corresponding labels, optional

        Returns:
            Either an array of cropped inputs or a tuple of two arrays of cropped inputs and labels, respectively
        """
        if self.cropped:
            warnings.warn(
                "Cropping twice could modify the cropper's state")

        self.n_original = images.shape[0]
        h, w = self.crop_shape
        step_y, step_x = (h, w) if self.step is None else (
            self.step, self.step)
        self.row_steps = (images.shape[1] - h)//step_y + 1
        self.col_steps = (images.shape[2] - w)//step_x + 1
        self.shape_uncropped = (images.shape[0], (self.row_steps - 1) * step_y + h,
                                (self.col_steps - 1) * step_x + w, labels.shape[3] if labels is not None else None)
        self.n_crops = self.n_original * self.row_steps * self.col_steps

        # calcular el punto de inicio según la esquina que estemos tomando como referencia
        self.start_y = 0 if self.top else images.shape[1] - self.shape_uncropped[1]
        self.start_x = 0 if self.left else images.shape[2] - self.shape_uncropped[2]

        imagenes_cropped = np.zeros((self.n_crops, h, w, *images.shape[3:]), np.uint8)
        if labels is not None:
            if labels.shape[0] != images.shape[0] or labels.shape[2] != images.shape[2] or labels.shape[2] != images.shape[2]:
                warnings.warn(f"El número de imágenes o su tamaño no coincide con el de las etiquetas:  {images.shape[0:3]} vs {labels.shape[0:3]}.")
            labels_cropped = np.zeros((self.n_crops, h, w, labels.shape[3]), np.uint8)

        for x in range(self.n_original):
            for i in range(self.row_steps):
                for j in range(self.col_steps):
                    c = x * self.row_steps * self.col_steps + i * self.col_steps + j
                    top = self.start_y + i * step_y
                    left = self.start_x + j * step_x
                    imagenes_cropped[c] = images[x, top:(top + h), left:(left + w)]
                    if labels is not None:
                        labels_cropped[c] = labels[x, top:(top + h), left:(left + w)]
                    c += 1

        images = imagenes_cropped
        if labels is not None:
            labels = labels_cropped

        self.cropped = True
        if labels is not None:
            return images, labels
        else:
            return images

    def decrop(self, predictions, reduce_f=lambda x, y: x + y, initial=0):
        """Label decropping method
        
        Puts back together label crops to form original image labels.

        Args:
            predictions: array of cropped labels
            reduce_f: function defining what to do with overlapping labels
            initial: initial value for reduce function

        Returns:
            Array of uncropped (reconstructed) labels

        """
        if not self.cropped:
            warnings.warn("No se puede deshacer un recorte sin hacerlo")
            return None
        if predictions.shape[0] != self.n_crops:
            warnings.warn(
                "No coinciden el número de predicciones con el de crops")
        if self.shape_uncropped[3] is None:
            # añadimos el número de clases a la shape
            self.shape_uncropped = (
                *self.shape_uncropped[0:3], predictions.shape[3])

        # creamos una matriz inicial de elementos neutros
        uncropped = np.float64(np.full(self.shape_uncropped, initial))
        # tamaño de cada crop
        h, w = self.crop_shape
        # paso hacia abajo y paso a la derecha
        step_y, step_x = (h, w) if self.step is None else (
            self.step, self.step)

        for x in range(self.n_original):
            for i in range(self.row_steps):
                for j in range(self.col_steps):
                    c = x * self.row_steps * self.col_steps + i * self.col_steps + j
                    uncropped[x, i*step_y:(i*step_y + h), j*step_x:(j*step_x + w)] = reduce_f(
                        uncropped[x, i*step_y:(i*step_y + h), j*step_x:(j*step_x + w)], predictions[c])

        return uncropped


def processInput(x,y,parameters,soft_labelling=False):
    applyFilters =      bool(parameters.get('applyfilters',True))
    clahe       =       bool(parameters.get("clahe",True))
    sigma       =       parameters.get("sigma",3)
    truncate    =       parameters.get("truncate",4)

    if applyFilters:
        x_ = gray_to_3Channel(x,clahe)
    else:
        x_ = gray_repeat3(x,clahe)
    x_ = np.swapaxes(x_,2,3)
    x_ = np.swapaxes(x_,1,2)
    # print(x_.shape)
    y_ = None
    if y is not None:
        y_ = np.swapaxes(y,2,3)
        y_ = np.swapaxes(y_,1,2)
        # print(y_.shape)
    if soft_labelling:
        for i in range(len(y_)):
            for j in range(len(y_[i])):
                #plt.imsave("No_Gaussiana.png",y_[i,j])
                y_[i,j] = gaussian_filter(y_[i,j],sigma,mode = "nearest",truncate=truncate)
                #plt.imsave("Gaussiana.png",y_[i,j])
                
        
    return x_, y_
    
        

def cropImages(images, labels, sizeImages=(224, 224)):
    return ImageCropper(sizeImages).crop(images, labels)

def apply_clahe(images):
    # CLAHE only plays nice with uint8 so we convert any float images
    # We assume that float pixels are in the range [0,1]
    use_float = images.dtype == np.float32
    if use_float:
        images = np.uint8(images * 255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(len(images)):
        images[i] = clahe.apply(images[i])
    # Convert back to float if needed
    if use_float:
        images = np.float32(images) / 255
    return images

def gray_repeat3(images, clahe=True):
    if clahe:
        images = apply_clahe(images)
    rgbimages = np.zeros((*images.shape, 3), images.dtype)
    rgbimages[:, :, :, 2] = rgbimages[:, :,:, 1] = rgbimages[:, :, :, 0] = images
    return rgbimages

def minmax(arr):
    m = np.min(arr)
    return (arr - m)/(np.max(arr) - m)

def gray_to_3Channel(images, clahe=True):
    if clahe:
        images = apply_clahe(images)
        
    rgbimages = np.zeros((*images.shape[0:3], 3), images.dtype)
    rgbimages[:, :, :, 0] = images
    use_float = images.dtype == np.float32
    
    for i in range(len(images)):
        # Sobel filter outputs in an unknown range almost centered in 0
        # so we obtain results in float, then normalize via minmax and
        # compute their sqrt( squared sum ), finally converting this back
        # onto our 0-255 range if needed
        sx = ndimage.sobel(images[i], axis=0, mode='constant', output=np.float32)
        sy = ndimage.sobel(images[i], axis=1, mode='constant', output=np.float32)
        # print("sx", np.min(sx), np.max(sx), sx.dtype)
        sx = minmax(sx)
        # print("sxmin", np.min(sx), np.max(sx), sx.dtype)
        sy = minmax(sy)
        sob = minmax(np.hypot(sx, sy))
        # print("hyp", np.min(sob), np.max(sob), sob.dtype)
        if not use_float:
            # print("sob", np.min(sob * 255), np.max(sob * 255))
            sob = (sob * 255).astype(np.uint8)
            # print("sobi", np.min(sob), np.max(sob), sob.dtype)
        rgbimages[i, :, :, 1] = sob
        # LBP plays nice with uint8 and float32
        rgbimages[i, :, :, 2] = LBP(images[i], 8, 1)
        
    return rgbimages

# Recive a Segmentation torch Dataset with elements (X,Y) where X is an image with 3,H,W dimension and 
# Y is a mask of C,H,W dimension

class mySegmentationAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labeled=True):
        self.dataset = dataset
        self.labeled = labeled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,i):
        image = self.dataset.__getitem__(i)
        if self.labeled:
            image, label = image

        if np.random.random() > 0.5:
            image = torchvision.transforms.functional.hflip(image)
            if self.labeled: label = torchvision.transforms.functional.hflip(label)
        if np.random.random() > 0.5:
            image = torchvision.transforms.functional.vflip(image)
            if self.labeled: label = torchvision.transforms.functional.vflip(label)

        rotation_angle = np.random.random()*180

        size = image.size()
        image = torchvision.transforms.functional.pad(image,[int(size[-2]/4)+1,int(size[-1]/4)+1],padding_mode="symmetric")
        if self.labeled: label = torchvision.transforms.functional.pad(label,[int(size[-2]/4)+1,int(size[-1]/4)+1],padding_mode="symmetric")

        image = torchvision.transforms.functional.rotate(image,rotation_angle,resample=PIL.Image.BILINEAR)
        if self.labeled: label = torchvision.transforms.functional.rotate(label,rotation_angle,resample=PIL.Image.NEAREST)
        
        image = torchvision.transforms.functional.center_crop(image,(size[-2],size[-1]))
        if self.labeled: label = torchvision.transforms.functional.center_crop(label,(size[-2],size[-1]))

        # plt.imsave("imagen0.png",image[0])
        # image = torchvision.transforms.functional.adjust_brightness(image,)
        # plt.imsave("imagen1.png",image[0])
        # exit(-1)

        return (image, label) if self.labeled else image


def applyDA(DS, labeled=True):
    return mySegmentationAugmentedDataset(DS, labeled)
