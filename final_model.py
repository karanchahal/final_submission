# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
# from .convlstm import ConvLSTM
from collections import OrderedDict


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out



def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")




class Decoder(nn.Module):
    def __init__(self, num_classes=3):
        super(Decoder, self).__init__()
        self.num_output_channels = num_classes
        self.num_ch_enc = self.num_ch_enc = np.array([32, 64, 128, 256, 512])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_concat = np.array([64, 128, 256, 512, 128])
        self.conv_mu = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv_log_sigma = nn.Conv2d(128, 128, 3, 1, 1)
        outputs = {}
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            num_ch_concat = self.num_ch_concat[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, 3, 1, 1) #Conv3x3(num_ch_in, num_ch_out)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] =  nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(num_ch_out, num_ch_out, 3, 1, 1) #ConvBlock(num_ch_out, num_ch_out)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))



    def forward(self, x, is_training=True):
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            #x = torch.cat((x, features[i-6]), 1)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)

        if is_training:
            x = self.convs["topview"](x) #self.softmax(self.convs["topview"](x))
        else:
            softmax = nn.Softmax2d()
            x = softmax(self.convs["topview"](x))

        return x


class FinalModel(nn.Module):

    def __init__(self):
        super(FinalModel, self).__init__()

        self.encoder = ResnetEncoder(num_layers=50, pretrained=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=512, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = Decoder()
        self.output_size =(800,800)


    def forward(self, images):


        features = []
        # num_images = 6
        for im in images:
            feat = F.relu(self.encoder(im)[-1])
            h = feat.size(2)
            w = feat.size(3)
            feat = feat.view(-1, h, w)
            features.append(feat)
        features = torch.stack(features)
        # bt_sz = x.size(0)
        features = F.relu(self.downsample(features))
        dec = self.decoder(features)
        final_out = F.interpolate(dec, self.output_size)

        return final_out

class SingleModel(nn.Module):

    def __init__(self):
        super(SingleModel, self).__init__()
        self.encoder = ResnetEncoder(num_layers=18, pretrained=False)
        self.upsampler_cls = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 10),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 10),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 15, 10),
            nn.ReLU(),
            nn.BatchNorm2d(15),
        )

        self.upsampler_box = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 10),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 10),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 15*4, 10),
            nn.ReLU(),
            nn.BatchNorm2d(15*4),
        )
        # self.classifier = F.upsample()
    
    def forward(self, x):
        bt_sz = x.size(0)
        x = self.encoder(x)[-1]
        pred_cls = self.upsampler_cls(x)
        pred_cls = F.interpolate(pred_cls, (40,40))

        pred_box = self.upsampler_box(x)
        pred_box = F.interpolate(pred_box, (40,40))

        pred_cls = pred_cls.view(bt_sz, -1)
        pred_box = pred_box.view(bt_sz, -1, 4)
        return pred_cls, pred_box



class GoodSegModel(nn.Module):

    def __init__(self):
        super(GoodSegModel, self).__init__()
        self.encoder = ResnetEncoder(num_layers=18, pretrained=False)
        self.conv1 = nn.Conv2d(512, 32, 1)
        self.upsampler_cls = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 10),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 10),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 2, 10),
        )

        # self.classifier = F.upsample()
    
    def forward(self, samples):
        finals = []
        for x in samples:
            bt_sz = x.size(0)
            feats = self.encoder(x)[-1]
            x = self.conv1(feats)
            x = x.view(-1, 8, 10)
            finals.append(x)

        finals = torch.stack(finals)

        x = self.upsampler_cls(finals)
        x = torch.softmax(F.interpolate(x, (800,800)), dim=1)
        return x


# model = SingleModel()
# a = torch.randn((6, 3, 256, 306))
# x = model(a)
# print(x.size())

# model = GoodSegModel()
# a = torch.randn((4, 6, 3, 256, 306))
# x = model(a)
# print(x.size())
