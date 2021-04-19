import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class PixelNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(PixelNet, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features)
        self.linear = nn.Sequential(
            nn.Linear(1472, 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, n_classes, bias=True),
        )
        self.is_train = False

    def set_train_flag(self, flag):
        self.is_train = flag

    def generate_rand_ind(self, labels, n_class, n_samples):
        n_samples_avg = int(n_samples / n_class)
        rand_ind = []
        for i in range(n_class):
            positions = np.where(labels.view(1, -1) == i)[1]
            if positions.size == 0:
                continue
            else:
                rand_ind.append(np.random.choice(positions, n_samples_avg))
        rand_ind = np.random.permutation(np.hstack(rand_ind))
        return rand_ind

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def forward(self, x):
        # take the feature maps before MaxPooling layers
        feature_maps_index = {3, 8, 15, 22, 29}
        if self.is_train:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in feature_maps_index:
                    upsample = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    features.append(upsample)
            outputs = torch.cat(features, 1)
            outputs = outputs.permute(0, 2, 1)
            outputs = self.linear(outputs)
            outputs = outputs.permute(0, 2, 1)
        else:
            size, n_pixels = (x.size(2), x.size(3)), x.size(2)*x.size(3)
            upsample_maps = []
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in feature_maps_index:
                    upsample = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
                    upsample_maps.append(upsample)
            # perform batch processing for fully connected layers to reduce memory
            outputs = []
            for ind in range(0, n_pixels, 10000):
                ind_range = range(ind, min(ind+10000, n_pixels))
                features = []
                for upsample in upsample_maps:
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:, :, ind_range]
                    features.append(upsample)
                output = torch.cat(features, 1)
                output = output.permute(0, 2, 1)
                output = self.linear(output)
                output = output.permute(0, 2, 1)
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
            outputs = outputs.reshape(*outputs.shape[:2], *size)
        return outputs

        



