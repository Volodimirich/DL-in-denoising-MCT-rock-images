#from skimage import color
#from skimage.util import random_noise
import pandas as pd
import torch.utils.data
from PIL import Image
import cv2
class Dataset:
    '''Noisy/filtered CT images dataset'''

    def __init__(self, paths, transform=None):
        """
        :param paths: paths to noisy and filtered CT images
        :param transform: transformation to be applied to images
        """

        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths['original'])

    def __getitem__(self, idx):
        o_img_path = self.paths['original'][idx]
        f_img_path = self.paths['filtered'][idx]

        o_img = cv2.imread(o_img_path, 0)
        f_img = cv2.imread(f_img_path, 0)
        
        #### FIX
        dim = (402, 402)

        #
        ## resize image
        o_img = cv2.resize(o_img, dim, interpolation=cv2.INTER_AREA)
        f_img = cv2.resize(f_img, dim, interpolation=cv2.INTER_AREA)

        sample = {'X': o_img, 'Y': f_img}

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = f_img_path.split('/')[-1]

        return sample


# class BSD:

    # def __init__(self, paths, sigma, transform=None):
    #
    #     self.paths = paths
    #     self.transform = transform
    #     self.sigma = sigma
    #
    # def __len__(self):
    #     return len(self.paths)
    #
    # def __getitem__(self, idx):
    #     pass
        #target_img = cv2.imread(self.paths[idx])
        #target_img = color.rgb2gray(target_img)
        #noisy_img = random_noise(target_img, mode='gaussian', var=self.sigma**2)


        #sample = {"X" : noisy_img,
                  #"Y" : target_img
                  #}

        #if self.transform:
            #sample = self.transform(sample)

        #return sample
