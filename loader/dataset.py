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
        dim = (402, 400)

        #
        ## resize image
        o_img = cv2.resize(o_img, dim, interpolation=cv2.INTER_AREA)
        f_img = cv2.resize(f_img, dim, interpolation=cv2.INTER_AREA)

        sample = {'X': o_img, 'Y': f_img}

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = f_img_path.split('/')[-1]

        return sample