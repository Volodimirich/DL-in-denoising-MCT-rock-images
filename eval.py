import glob

from matplotlib import pyplot as plt
from losses.metrics import PSNR

from losses.losses import *
from src.utils import get_config, parse_args
from loader.data_factory import make_ct_datasets
from models.denoising.rednet import RED_Net_20
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def eval(model, loader, device):

    loss_list = []
    for sample in loader:
        X, Y_true = sample['X'], sample['Y']

        # transfer tensors to the current device
        X = X.to(device)
        Y_true = Y_true.to(device)


        # forward propagate
        with torch.no_grad():
            Y_pred = model(X)
            Y_pred = torch.nan_to_num(Y_pred)

        # loss_list.append(compare_psnr(Y_pred.detach().cpu().numpy()[0], Y_true.detach().cpu().numpy()[0]))
        loss_list.append(PSNR(Y_pred.detach().cpu().numpy()[0], Y_true.detach().cpu().numpy()[0]))
    result_psnr = sum(loss_list)/len(loss_list)
    print(result_psnr)

    return result_psnr, loss_list

if __name__ == '__main__':
    args = parse_args()
    print('args', args)
    paths, configs = get_config(args.paths), get_config(args.config)

    # model = RED_Net_20(inp=1, out=64,kernel=3,st=1,pad=1).to(device)
    print(f'Current device: {device}')
    weights_path = [file for file in glob.iglob('trained-models/*.pt')]
    print(weights_path)

    model = RED_Net_20()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    ####### EVAL ######
    for wt in weights_path:
        print(wt)
        model.load_state_dict(torch.load(wt)['model_state_dict'])
        loader, _ = make_ct_datasets(configs, paths, crop_size=256, TRAIN_SIZE=1)
        score = eval(model, loader,  device)
        print('___________')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss function')
    plt.savefig('crop_plot_c.png')
