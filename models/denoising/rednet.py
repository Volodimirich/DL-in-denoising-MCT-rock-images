import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


class RED_Net_30(nn.Module):
    """
    This baseline is 30-layered residual encoder-decoder neural network
    with symmetric skip-connections between convolutional and deconvolutional
    layers with step 2, ReLU activations, filters of constant size 3x3, constant
    number of channels (128) in activations of each layer, padding = 1, stride = 1,
    no max-pooling.
    """

    def __init__(self, num_rep, inp=1, out=128, kernel=3, st=1, pad=3):
        super(RED_Net_30, self).__init__()
        print('30?')

        self.conv_entry = nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=kernel, stride=st, padding=pad),
                                        nn.ReLU())
        self.conv_rep = nn.Sequential(nn.Conv2d(in_channels=out, out_channels=out, kernel_size=kernel, stride=st, padding=pad),
                                      nn.ReLU())
        self.deconv_rep = nn.Sequential(nn.ConvTranspose2d(in_channels=out, out_channels=out, kernel_size=kernel, stride=st,
                                             padding=pad), nn.ReLU())
        self.deconv_exit = nn.ConvTranspose2d(in_channels=out, out_channels=inp, kernel_size=kernel, stride=st,
                                              padding=pad)
        self.num_rep = num_rep

    def forward(self, X):
        val_dict = {}
        x_rep = self.conv_entry(X)

        for i in range(1, self.num_rep):
            x_rep = self.conv_rep(x_rep)
            if not i % 2:
                val_dict[i] = x_rep

        for i in range(self.num_rep, 1, -1):
            val = x_rep + val_dict[i] if not i % 2 else x_rep
            x_rep = self.deconv_rep(val)
        X = self.deconv_exit(x_rep)

        return X

class RED_Net_20(nn.Module):
    """
    This baseline is 20-layered residual encoder-decoder neural network
    with symmetric skip-connections between convolutional and deconvolutional
    layers with step 2, ReLU activations, filters of constant size 3x3, constant
    number of channels (128) in activations of each layer, padding = 1, stride = 1,
    no max-pooling.
    """

    def __init__(self):
        super(RED_Net_20, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_10 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_6 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_7 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_8 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_9 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_10 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.conv_1(X)
        X = self.conv_2(X)

        X_3 = self.conv_3(X)
        X = self.conv_4(X_3)

        X_5 = self.conv_5(X)
        X = self.conv_6(X_5)

        X_7 = self.conv_7(X)

        X = self.conv_8(X_7)
        X_9 = self.conv_9(X)

        X = self.conv_10(X_9)
        X = self.deconv_1(X)

        X = self.deconv_2(F.relu(X + X_9))
        X = self.deconv_3(X)

        X = self.deconv_4(F.relu(X + X_7))
        X = self.deconv_5(X)

        X = self.deconv_6(F.relu(X + X_5))
        X = self.deconv_7(X)

        X = self.deconv_8(F.relu(X + X_3))
        X = self.deconv_9(X)

        X = self.deconv_10(X)

        return X
class Conv_block(torch.nn.Module):
    def __init__(self, inp, out, kernel=3, st=1, pad=1, batch=64):
        super(Conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=kernel, stride=st, padding=pad),
            nn.BatchNorm2d(batch),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class Deconv_block(torch.nn.Module):
    def __init__(self, out, kernel=3, st=1, pad=1, batch=64):
        super(Deconv_block, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out, out_channels=out, kernel_size=kernel, stride=st, padding=pad),
            nn.BatchNorm2d(batch),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.deconv(x)