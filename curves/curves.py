import os
import pydoc

from matplotlib import pyplot as plt
from torch import nn

from losses.losses import SSIMLoss, PerceptualLoss
from src.utils import get_config, parse_args, Logger, save_model
from loader.data_factory import make_ct_datasets
from models.denoising.rednet import RED_Net_20
import torch
import pickle

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
            save_model(model, os.path.join(weights_path, model_name))
            best_loss = loss_logger.average

        # save checkpoint
        save_model(model, os.path.join(weights_path, 'checkpoint.pt'))

    return train_loss_list, val_loss_list

def curve_a():
    args = parse_args()
    print('args', args)
    paths, configs = get_config(args.paths),  get_config(args.config)


    # model = RED_Net_20(inp=1, out=64,kernel=3,st=1,pad=1).to(device)
    print(f'Current device: {device}')

    # try:
    #     pretrained = configs['train_params']['pretrained']
    #     if pretrained:
    #         model_dumps = torch.load(configs['train_params']['path_weights'], map_location=device)
    #         model.load_state_dict(model_dumps['model_state_dict'])
    #         print(f'Weights loaded from model {configs["train_params"]["path_weights"]}')
    # except KeyError:
    #     print('A parameter wasn`t found in the config file')

    ####### OPTIMIZER ######
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ####### SCHEDULER ######
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True,
    #                                                        min_lr=0.0000001)
    ####### CRITERION ######
    # loss = torch.nn.MSELoss()
    loss = SSIMLoss()

    ####### TRAINING ######
    max_epoch = int(configs['train_params']['max_epoch'])
    x = [i for i in range(1, 51)]
    for col, crop in zip(['r', 'g', 'b'], [128, 256, 400]):
        # model = RED_Net_20(inp=1, out=64, kernel=3, st=1, pad=1)
        model = RED_Net_20()
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model.to(device)
        ####### OPTIMIZER ######
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ####### SCHEDULER ######
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True,
                                                               min_lr=0.0000001)


        train_loader, val_loader = make_ct_datasets(configs, paths, crop_size=crop)
        loss_list, val_list = train(model, optimizer, loss, train_loader, max_epoch, device, val_loader,
              scheduler=scheduler, weights_path=paths['dumps']['weights'], model_name='Resnet 20')
        plt.plot(x, loss_list, label=f'{crop}x{crop}_train', color=col, linestyle='-')
        plt.plot(x, val_list, label=f'{crop}x{crop}_val', color=col, linestyle='--')
        del model
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        with open(f'{crop}x{crop}_list_loss.pkl', 'wb') as f:
            pickle.dump(loss_list, f)

        with open(f'{crop}x{crop}_list_loss_val.pkl', 'wb') as f:
            pickle.dump(val_list, f)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss function')
    plt.savefig('crop_plot.png')

def curve_b():
    args = parse_args()
    print('args', args)
    paths, configs = get_config(args.paths),  get_config(args.config)


    # model = RED_Net_20(inp=1, out=64,kernel=3,st=1,pad=1).to(device)
    print(f'Current device: {device}')

    loss = PerceptualLoss()

    ####### TRAINING ######
    max_epoch = int(configs['train_params']['max_epoch'])
    x = [i for i in range(1, 51)]
    for col, crop in zip(['r', 'g', 'b'], [128, 256]):
        # model = RED_Net_20(inp=1, out=64, kernel=3, st=1, pad=1)
        model = RED_Net_20()
        model = nn.DataParallel(model, device_ids=[0,1])
        model.to(device)
        ####### OPTIMIZER ######
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ####### SCHEDULER ######
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True,
                                                               min_lr=0.0000001)


        train_loader, val_loader = make_ct_datasets(configs, paths, crop_size=crop)
        loss_list, val_list = train(model, optimizer, loss, train_loader, max_epoch, device, val_loader,
              scheduler=scheduler, weights_path=paths['dumps']['weights'], model_name='Resnet 20')
        plt.plot(x, loss_list, label=f'{crop}x{crop}_train', color=col, linestyle='-')
        plt.plot(x, val_list, label=f'{crop}x{crop}_val', color=col, linestyle='--')
        del model
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        with open(f'{crop}x{crop}_list_loss.pkl', 'wb') as f:
            pickle.dump(loss_list, f)

        with open(f'{crop}x{crop}_list_loss_val.pkl', 'wb') as f:
            pickle.dump(val_list, f)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss function')
    plt.savefig('crop_plot_b.png')
