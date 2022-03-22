import os

import piq
from matplotlib import pyplot as plt

from losses.losses import *
from src.utils import get_config, parse_args, Logger, save_model
from loader.data_factory import make_ct_datasets
from models.denoising.DnCNN import DnCNN
from models.denoising.rednet import RED_Net_20
import torch
import pickle

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, train_loader, num_epoch, device, val_loader=None,
          scheduler=None, save_best=True, weights_path='', model_name='best_model.pt'):
    """
     Starts training process of the input model, using specified optimizer

    :param model: torch model
    :param optimizer: torch optimizer
    :param criterion: torch criterion
    :param train_loader: torch dataloader instance of training set
    :param val_loader: torch dataloader instance of validation set
    :param num_epoch: number of epochs to train
    :param device: device to train on
    """

    loss_logger = Logger()
    best_loss = float('inf')
    train_loss_list, val_loss_list = [], []


    for epoch in range(num_epoch):
        model.train()
        loss_logger.reset()
        for sample in train_loader:
            X, Y_true = sample['X'], sample['Y']

            # transfer tensors to the current device
            X = X.to(device)
            Y_true = Y_true.to(device)


            # zero all gradients
            optimizer.zero_grad()

            # forward propagate
            Y_pred = model(X)

            Y_pred = torch.nan_to_num(Y_pred)

            loss = criterion(Y_pred, Y_true)

            loss_logger.update(loss.item())

            # backprop and update the params
            loss.backward()
            optimizer.step()

        train_loss_list.append(loss_logger.average)
        print(f"Epoch: {epoch} | Train loss: {loss_logger.average} |", end=" ")

        # evaluation of model performance on validation set
        loss_logger.reset()
        model.eval()
        for sample in val_loader:
            X = sample['X'].to(device)
            Y_true = sample['Y'].to(device)


            with torch.no_grad():
                Y_pred = model(X)
                Y_pred = torch.nan_to_num(Y_pred)
                val_loss = criterion(Y_pred, Y_true)
            loss_logger.update(val_loss.item())


        print(f"Val loss: {loss_logger.average}")
        val_loss_list.append(loss_logger.average)

        # scheduler
        if scheduler:
            scheduler.step(loss_logger.average)

        # save the best model
        if loss_logger.average < best_loss and save_best:
            # save_model(model, os.path.join(weights_path, model_name))
            best_loss = loss_logger.average
        save_model(model, os.path.join(weights_path, model_name))

    return train_loss_list, val_loss_list

def loss_test():
    args = parse_args()
    print('args', args)
    paths, configs = get_config(args.paths), get_config(args.config)

    # model = RED_Net_20(inp=1, out=64,kernel=3,st=1,pad=1).to(device)
    print(f'Current device: {device}')
    sr_sim = piq.SRSIMLoss(data_range=1.)

    # Losses which we try to use in this task
    # f_sim = piq.FSIMLoss(data_range=1., reduction='none')
    # gmsd_loss = piq.GMSDLoss(data_range=1., reduction='none')
    # haarpsi_index = piq.HaarPSILoss(data_range=1., reduction='none')
    # lpips_loss = piq.LPIPS(reduction='none')
    # mdsi_loss = piq.MDSILoss(data_range=1., reduction='none')
    # ms_ssim_loss = piq.MultiScaleSSIMLoss(data_range=1., reduction='none')
    # ms_gmsd_loss = piq.MultiScaleGMSDLoss(chromatic=False, data_range=1., reduction='none')
    # pieapp_loss = piq.PieAPP(reduction='none', stride=32)
    # tv_loss = piq.TVLoss(reduction='none')
    # vsi_loss = piq.VSILoss(data_range=1.)

    loss_list = [sr_sim, nn.L1Loss(), nn.MSELoss(), SSIMLoss(), PerceptualLoss(), PerceptualLoss31(), PerceptualLoss34()
                 ]
    names = ['SR_Sim', 'L1Loss', 'MSELoss','SSIM', 'VGG11', 'VGG31', 'VGG34']

    ####### TRAINING ######
    max_epoch = int(configs['train_params']['max_epoch'])
    x = [i for i in range(1, 151)]
    for loss, name in zip(loss_list, names):
        model = RED_Net_20()
        # model = DnCNN(channels=1)
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True,
                                                               min_lr=0.0000001)

        train_loader, val_loader = make_ct_datasets(configs, paths, crop_size=128)
        loss_list, val_list = train(model, optimizer, loss, train_loader, max_epoch, device, val_loader,
                                    scheduler=scheduler, weights_path=paths['dumps']['weights'],
                                    model_name=f'{name}_RedNet.pt')

        del model
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        with open(f'{name}_list_loss.pkl', 'wb') as f:
            pickle.dump(loss_list, f)

        with open(f'{name}_list_loss_val.pkl', 'wb') as f:
            pickle.dump(val_list, f)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss function')
    plt.savefig('crop_plot_c.png')

def solo_learn():
    args = parse_args()
    print('args', args)
    paths, configs = get_config(args.paths), get_config(args.config)

    # model = RED_Net_20(inp=1, out=64,kernel=3,st=1,pad=1).to(device)
    print(f'Current device: {device}')

    loss = torch.nn.MSELoss()

    ####### TRAINING ######
    name = 'Linear_long'
    max_epoch = int(configs['train_params']['max_epoch'])
    x = [i for i in range(1, 51)]
    # model = RED_Net_20()
    model = DnCNN(channels=1)
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True,
                                                           min_lr=0.0000001)

    train_loader, val_loader = make_ct_datasets(configs, paths, crop_size=128)
    train(model, optimizer, loss, train_loader, max_epoch, device, val_loader,
                                scheduler=scheduler, weights_path=paths['dumps']['weights'], model_name=f'{name}.pt')

if __name__ == '__main__':
    loss_test()