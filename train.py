import os
import pydoc

import piq
from matplotlib import pyplot as plt
from torch import nn

from losses.losses import *
from src.utils import get_config, parse_args, Logger, save_model
from loader.data_factory import make_ct_datasets
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

if __name__ == '__main__':
    args = parse_args()
    print('args', args)
    paths, configs = get_config(args.paths),  get_config(args.config)

    # model = RED_Net_20(inp=1, out=64,kernel=3,st=1,pad=1).to(device)
    print(f'Current device: {device}')
    sr_sim =  piq.SRSIMLoss(data_range=1.)

    loss_list = [nn.L1Loss(), nn.MSELoss(), SSIMLoss(), PerceptualLoss(), PerceptualLoss31(), PerceptualLoss34(),
                 sr_sim]
    colors = ['r', 'g', 'b','orange', 'cyan', 'pink', 'silver']
    names = ['L1_loss', 'L2_loss', 'SSIM_loss', 'VGG11_loss', 'VGG31_loss', 'VGG34_loss', 'SRSIM_loss']
    ####### TRAINING ######
    max_epoch = int(configs['train_params']['max_epoch'])
    x = [i for i in range(1, 51)]
    for col, loss, name in zip(colors, loss_list, names):
        model = RED_Net_20()
        model = nn.DataParallel(model, device_ids=[0,1])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True,
                                                               min_lr=0.0000001)

        train_loader, val_loader = make_ct_datasets(configs, paths, crop_size=128)
        loss_list, val_list = train(model, optimizer, loss, train_loader, max_epoch, device, val_loader,
              scheduler=scheduler, weights_path=paths['dumps']['weights'], model_name=f'{name}.pt')

        plt.plot(x, loss_list, label=f'{name}_train', color=col, linestyle='-')
        plt.plot(x, val_list, label=f'{name}_val', color=col, linestyle='--')
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
