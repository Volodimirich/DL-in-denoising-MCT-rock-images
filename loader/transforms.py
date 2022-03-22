import numpy as np
import torch

class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']

        h, w = X.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        X = X[top: top + new_h,
             left: left + new_w]

        Y = Y[top: top + new_h,
             left: left + new_w]

        return {'X': X,
                'Y': Y}

class ToFloat:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']

        X = X / 255
        Y = Y / 255

        return {'X': X,
                'Y': Y}

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']

        if len(X.shape) == 2:
            X = X[:, :, None]
            Y = Y[:, :, None]

        X = X.transpose((2, 0, 1))
        Y = Y.transpose((2, 0, 1))
        return {'X': torch.tensor(X, dtype=torch.float32),
                'Y': torch.tensor(Y, dtype=torch.float32)}

