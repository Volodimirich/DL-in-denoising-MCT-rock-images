import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

class GeneralFourier2d(torch.nn.Module):
    def __init__(self, image_size, log=False):
        super(GeneralFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='W1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        self.register_parameter(name='B1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='W2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='B2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        torch.nn.init.ones_(self.W1)
        torch.nn.init.zeros_(self.B1)
        torch.nn.init.ones_(self.W2)
        torch.nn.init.zeros_(self.B2)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        w1 = torch.nn.ReLU()(self.W1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        w2 = torch.nn.ReLU()(self.W2.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b1 = torch.nn.ReLU()(self.B1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b2 = torch.nn.ReLU()(self.B2.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.fft.rfft(x, dim=3, norm='forward')
        print('rft_x', rft_x.shape)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))
        print('init_spectrum', init_spectrum.shape)


        if self.log:
            spectrum = w2 * self.activation(w1 * torch.log(1 + init_spectrum) + b1) + b2
        else:
            spectrum = w2 * self.activation(abs(w1 * init_spectrum + b1)) + b2

        irf = torch.fft.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          dim=3, norm='forward')

        return irf


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
        # self.fl = GeneralFourier2d((1, 128, 254), log=False)

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
        # X = self.fl(X)
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
