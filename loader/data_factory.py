import numpy as np
from glob import glob
import os
from loader.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from loader.transforms import RandomCrop, ToFloat, ToTensor

def make_ct_datasets(configs, paths, crop_size = None, TRAIN_SIZE = 0.9):
    o_img_paths = np.array(sorted(glob(os.path.join(paths['data']['path'], 'Original/*'))))
    f_img_paths = np.array(sorted(glob(os.path.join(paths['data']['path'], 'Filtered/*'))))
    img_paths_train = {'original': o_img_paths[:int(TRAIN_SIZE * len(o_img_paths))],
                       'filtered': f_img_paths[:int(TRAIN_SIZE * len(f_img_paths))]
                      }
    img_paths_val = {'original': o_img_paths[int(TRAIN_SIZE * len(o_img_paths)):],
                     'filtered': f_img_paths[int(TRAIN_SIZE * len(f_img_paths)):]
                    }

    if not crop_size:
        crop_size = configs['data_params']['augmentation_params']['crop_size']
    transforms_train = Compose([
                                RandomCrop(crop_size),
                                ToFloat(),
                                ToTensor()]
                              )
    transforms_val = Compose([
                              RandomCrop(crop_size),
                              ToFloat(),
                              ToTensor()]
                            )

    train_loader = DataLoader(Dataset(img_paths_train, transforms_train),
                              batch_size=configs['data_params']['batch_size'],
                              num_workers=configs['data_params']['num_workers'],
                              shuffle=True
                              )

    val_loader = DataLoader(Dataset(img_paths_val, transforms_val),
                            batch_size=1,
                            num_workers=configs['data_params']['num_workers'],
                            shuffle=False
                            )

    return train_loader, val_loader