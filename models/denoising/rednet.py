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
    def __init__(self, num_rep=9, inp=1, out=64, kernel=3, st=1, pad=3):
        super(RED_Net_20, self).__init__()
        self.conv_entry = nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=kernel, stride=st, padding=pad),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_exit = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out, out_channels=inp, kernel_size=kernel, stride=st, padding=pad),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.num_rep = num_rep

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.conv_sum = nn.ModuleList(Conv_block(inp=64, out=64).to(device) for _ in range(num_rep))
        self.deconv_sum = nn.ModuleList(Deconv_block(out=64).to(device) for _ in range(num_rep))
        print(self.deconv_sum[0] is self.deconv_sum[1])
        self.skip_positions = [1, 3, 5, 7]

    def forward(self, X):
        val_dict = {}
        x_rep = self.conv_entry(X)

        for i, conv_layer in zip(range(0, self.num_rep), self.conv_sum):
            x_rep = conv_layer(x_rep)
            if i in self.skip_positions:
                val_dict[i] = x_rep

        for i, deconv_layer in zip(range(self.num_rep-1, -1, -1), self.deconv_sum):
            val = x_rep + val_dict[i] if i in self.skip_positions else x_rep
            x_rep = deconv_layer(val)

        X = self.deconv_exit(x_rep)

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