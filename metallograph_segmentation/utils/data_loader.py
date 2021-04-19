import os
import glob
from collections import namedtuple
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy import ndimage
from skimage.feature import local_binary_pattern as LBP
from PIL import Image
import warnings
import json
import cv2

class Dataset(object):
    def __init__(self, path, cropbar=None, num_classes=4, image_size=(1280, 900), image_resizing=Image.BILINEAR, load_labels=True):
        self.path = path
        self.cropbar = cropbar
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_resizing = image_resizing
        self.load_labels = load_labels
    
    def remove_cropbar(self, image, custom_cropbar=None):
        """Remove micron bar from bottom of image"""
        if custom_cropbar is not None:
            if custom_cropbar > 0:
                return image[:-custom_cropbar]
            else:
                return image
        elif self.cropbar is not None and self.cropbar > 0:
            return image[:-self.cropbar]
        return image

    def data_from_raw(self, images_raw, labels_raw):
        # Image and label resizing (nearest value for class and bilinear for image)
        images = []
        for i in range(len(images_raw)):
            images.append(np.array(Image.fromarray(
                images_raw[i], mode="L").resize(self.image_size, self.image_resizing)))
        images = np.array(images)

        labels = []
        if self.load_labels:
            if len(labels_raw) > 0:
                for i in range(len(labels_raw)):
                    imageLabels = np.array(labels_raw[i])
                    imageLabels = imageLabels.astype(np.uint8) #?
                    imageLabels = Image.fromarray(imageLabels, mode="L")
                    imageLabels = imageLabels.resize(self.image_size, Image.NEAREST)
                    imageLabels = np.array(imageLabels)
                    if len(imageLabels.shape) > 2:
                        print(imageLabels.shape)
                        print("ERROR!")

                    labels.append(imageLabels)
                labels = np.array(labels) - np.min(labels)
                labels = to_categorical(labels, self.num_classes)

        return images, labels
    
    def data(self):
        raise NotImplementedError
    
    def partitions(self):
        raise NotImplementedError

    def folds(self):
        raise NotImplementedError

class H5Dataset(Dataset):
    def __init__(self, path, cropbar=None, num_classes=4, image_size=(1280, 900), image_resizing=Image.BILINEAR):
        super().__init__(path, cropbar, num_classes, image_size, image_resizing)
        self.exclude = set()
    

    def load_record(self, f, key):
        micrograph = f[key]
        im = self.remove_cropbar(micrograph['image'][...])
        l = self.remove_cropbar(micrograph['labels'][...])

        return im, l


    def reader(self):
        """ load uhcsseg training data from hdf5 """
        images, labels, names = [], [], []
        with h5py.File(self.path, 'r') as f:
            for key in f:
                if key in self.exclude:
                    continue
                im, l = self.load_record(f, key)
                names.append(key)
                images.append(im)
                labels.append(l)

        return np.array(images), np.array(labels), np.array(names)
    
    def data(self):
        images_raw, labels_raw, _ = self.reader()
        images, labels = self.data_from_raw(images_raw, labels_raw)
        return images, labels

class UHCSDataset(H5Dataset):
    """
    Public UHCS dataset from uhcs.h5
    """
    def __init__(self, path, name="uhcs", image_size=(1280, 900), image_resizing=Image.BILINEAR,num_classes=5):
        super().__init__(os.path.join(path, name + ".h5"), 38, 5, image_size, image_resizing)
        
        # throw out these micrographs for the spheroidite task
        # the input has a weird intensity distribution
        self.exclude = {
            '800C-85H-Q-4',
            '800C-8H-Q-2',
            '800C-90M-Q-1'
        }
    
    def data(self):
        images, labels = super().data()
        labels = labels[:, :, :, 0:4]
        return images, labels
    
    def partitions(self, seed=12345):
        images, labels = super().data()
        n = images.shape[0]
        np.random.seed(seed=seed)
        test = np.random.randint(0, n, size=np.int(np.ceil(0.2 * n)))
        return (
            (np.delete(images, test, axis=0),
            np.delete(labels, test, axis=0)),
            (images[test], labels[test]))

class AMDataset(H5Dataset):
    """
    Old ArcelorMittal dataset from arcelor.h5
    """
    def __init__(self, path, name="arcelor", image_size=(1280, 900), image_resizing=Image.BILINEAR):
        super().__init__(os.path.join(path, name + ".h5"), 0, 4, image_size, image_resizing)

    def partitions(self, test_index=11):
        images, labels = self.data()
        test_image = images[[test_index]]
        test_label = labels[[test_index]]

        train_images = np.delete(images, test_index, axis=0)
        train_labels = np.delete(labels, test_index, axis=0)

        return (
            # shape: (12, 900, 1280)
            train_images,
            # shape: (12, 900, 1280, 4)
            train_labels), (
            # shape: (1, 900, 1280)
            test_image,
            # shape: (1, 900, 1280, 4)
            test_label
        )

class V1Dataset(H5Dataset):
    """
    New ArcelorMittal dataset from V1Train.h5 and V1Test.h5
    """
    def __init__(self, path, train_name="V1Train", test_name="V1Test", image_size=(1280, 900), image_resizing=Image.BILINEAR):
        super().__init__(os.path.join(path, train_name + ".h5"), 0, 4, image_size, image_resizing)
        self.train_name = train_name
        self.test_name = test_name

    def partitions(self):
        train_images, train_labels = self.data()
        pathTest = os.path.sep.join(self.path.split(os.path.sep)[:-1])
        self.path = os.path.join(pathTest, self.test_name + ".h5")
        test_image, test_label = self.data()

        return (
            # shape: (12, 900, 1280)
            train_images,
            # shape: (12, 900, 1280, 4)
            train_labels), (
            # shape: (1, 900, 1280)
            test_image,
            # shape: (1, 900, 1280, 4)
            test_label
        )

class ImageDataset(Dataset):
    def __init__(self, path, cropbar=None, num_classes=4, image_size=(1280, 900), image_resizing=Image.BILINEAR,
                 microstructures_dir="microstructures", labels_dir="labels", label_cropbar=True, load_labels=True):
        super().__init__(path, cropbar, num_classes, image_size, image_resizing, load_labels=load_labels)
        self.images_root = os.path.join(path, microstructures_dir)
        self.labels_root = os.path.join(path, labels_dir)
        self.image_files = sorted(os.listdir(self.images_root))
        self.label_files = sorted(os.listdir(self.labels_root))
        self.label_cropbar = label_cropbar
    
    def reader(self):
        images_raw = {}
        labels_raw = {}
        # open(os.path.join(self.images_root, "all.txt"), "r").readlines()
        for img in self.image_files:
            images_raw[img] = self.remove_cropbar(cv2.imread(os.path.join(self.images_root, img.strip()), cv2.IMREAD_GRAYSCALE))

            # np.uint8(img_to_array(load_img(os.path.join(
            #     self.images_root, img.strip()), color_mode="grayscale"))[:, :, 0])

        if self.labels_root is not None:
            for img in self.label_files:
                labels_raw[img] = self.remove_cropbar(cv2.imread(os.path.join(self.labels_root, img.strip()), cv2.IMREAD_GRAYSCALE), None if self.label_cropbar else 0)
                # np.uint8(img_to_array(load_img(os.path.join(
                #     self.labels_root, img.strip()), color_mode="grayscale"))[:, :, 0])

        return images_raw, labels_raw

    def data(self):
        images_raw, labels_raw = self.reader()
        images, labels = self.data_from_raw(list(images_raw.values()), list(labels_raw.values()))
        return images, labels

class AdditiveDataset(ImageDataset):
    """Images from 'Additive_unlabeled'"""

    def __init__(self, path="data/Additive_unlabeled", cropbar=0, image_size=(1024, 768), image_resizing=Image.BILINEAR):
        super().__init__(path, cropbar, 0, image_size, image_resizing, "", "")

        self.image_files = [os.path.basename(i) for i in glob.glob(os.path.join(path, "*.tif"))]
        self.label_files = None
        self.labels_root = None # don't attempt to read labels

class MetalDAMDataset(ImageDataset):
    """Images from 'MetalDAM'"""

    def __init__(self, path="data/MetalDAM", cropbar=65, num_classes=5, image_size=(1024, 768),
                 image_resizing=Image.BILINEAR, microstructures_dir="images", labels_dir="labels", load_labels=True, crossval=True):
        super().__init__(path, cropbar, num_classes, image_size, image_resizing,
                         microstructures_dir, labels_dir, label_cropbar=False, load_labels=load_labels)
        self.apply_crossval = crossval
        if self.apply_crossval:
            rng = np.random.default_rng(12345678)
            test_selection = rng.choice(len(self.image_files), int(np.floor(0.2 * len(self.image_files))), replace=False)
            self.train_images = np.delete(np.array(self.image_files), test_selection)
            self.test_images = np.array(self.image_files)[test_selection]
            self.train_labels = np.delete(np.array(self.label_files), test_selection)
            self.test_labels = np.array(self.label_files)[test_selection]
        else:
            self.train_file = os.path.join(self.path, "train.txt")
            self.val_file = os.path.join(self.path, "val.txt")
            self.test_file = os.path.join(self.path, "test.txt")
            with open(self.train_file) as f:
                train_paths = f.read().splitlines()
            self.train_images = [lbl_name.split('.')[0]+".jpg" for lbl_name in train_paths]
            self.train_labels = [lbl_name.split('.')[0]+".png" for lbl_name in train_paths]
            with open(self.val_file) as f:
                val_paths = f.read().splitlines()
            self.val_images = [lbl_name.split('.')[0]+".jpg" for lbl_name in val_paths]
            self.val_labels = [lbl_name.split('.')[0]+".png" for lbl_name in val_paths]
            with open(self.test_file) as f:
                self.test_paths = f.read().splitlines()
            self.test_images = [lbl_name.split('.')[0]+".jpg" for lbl_name in self.test_paths]
            self.test_labels = [lbl_name.split('.')[0]+".png" for lbl_name in self.test_paths]

    def partitions(self):
        if not self.apply_crossval:
            images, labels = self.data()
            # couple images and labels with label filenames, for filtering into train and test subsets
            images = dict(zip(self.label_files, images))
            labels = dict(zip(self.label_files, labels))
            x_train = np.array([images[k] for k in self.train_labels])
            x_val = np.array([images[k] for k in self.val_labels])
            x_test = np.array([images[k] for k in self.test_labels])
            y_train = np.array([labels[k] for k in self.train_labels])
            y_val = np.array([labels[k] for k in self.val_labels])
            y_test = np.array([labels[k] for k in self.test_labels])

            return (x_train, y_train), (x_val, y_val), (x_test, y_test)
        else:
            super(ImageDataset, self).partitions()

DATASETS = {
    "uhcs": UHCSDataset,
    "additive": AdditiveDataset,
    "metaldam": MetalDAMDataset
}

def load_dataset(dataset="metaldam", **kwargs):
    return DATASETS[dataset](**kwargs).data()

def load_partitions(dataset="metaldam", **kwargs):
    return DATASETS[dataset](**kwargs).partitions()

def load_folds(dataset="metaldam", **kwargs):
    return DATASETS[dataset](**kwargs).folds()
