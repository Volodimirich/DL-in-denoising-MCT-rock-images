import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19

def gaussian_kernel(size, sigma):

    start = size // 2
    end = start
    if size % 2 == 1:
        end += 1

    arr = torch.arange(-start, end, dtype=torch.float32)
    xx, yy = torch.meshgrid(arr, arr)
    kernel = torch.exp(-0.5*(xx**2 + yy**2) / sigma**2)
    kernel /= kernel.sum()

    return kernel[None, None, :, :]


def ssim(pred_img, ref_img, sigma=1.5, kernel_size=None, return_cs=False, average=True):
    """
    :param pred_img: predicted image, torch.tensor of shape (batch_size, num_channels, width, height)
    :param ref_img: reference image (without noise), torch.tensor of shape (batch_size, num_channels, width, height)
    :return: similarity score, torch.float32
    """
    if len(pred_img.shape) != 4:
        raise ValueError("Image must have (batch, channel, height, width) dimensions ")

    k1 = 0.01**2
    k2 = 0.03**2
    num_ch = ref_img.shape[1]
    device = pred_img.device

    if not kernel_size:
        r = int(3.5 * sigma + 0.5)
        kernel_size = 2 * r + 1

    if kernel_size % 2 != 1:
        raise ValueError(f'Kernel size must be odd, but {kernel_size} found')

    kernel = gaussian_kernel(kernel_size, sigma).to(device)
    kernel = kernel.repeat(1, num_ch, 1, 1)

    mu_x = F.conv2d(pred_img, kernel)
    mu_y = F.conv2d(ref_img, kernel)

    sigma_x_sq = F.conv2d(pred_img * pred_img, kernel) - torch.pow(mu_x, 2)
    sigma_y_sq = F.conv2d(ref_img * ref_img, kernel) - torch.pow(mu_y, 2)
    sigma_xy = F.conv2d(pred_img * ref_img, kernel) - mu_x * mu_y

    cs = (2*sigma_xy + k2) / (sigma_x_sq + sigma_y_sq + k2)
    sim_score = cs * (2*mu_x*mu_y + k1) / (mu_x**2 + mu_y**2 + k1)

    if average:
        sim_score = sim_score.mean()
        cs = cs.mean()
    else:
        sim_score = sim_score.mean(dim=(1, 2, 3))
        cs = cs.mean(dim=(1, 2, 3))

    if return_cs:
        return sim_score, cs
    else:
        return sim_score


class SSIMLoss(nn.Module):

    def __init__(self, sigma=1.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, pred_img, ref_img):
        return 1.0 - ssim(pred_img, ref_img, self.sigma)

class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_calculator = list(vgg19(pretrained=True).to('cuda').children())[0][:4] #[:12] #[:18]

        for param in self.loss_calculator.parameters():
            param.requires_grad = False

    def forward(self, pred_img, ref_img):
        self.loss_calculator.eval()
        pred_img = self.loss_calculator(pred_img.repeat(1, 3, 1, 1))
        ref_img = self.loss_calculator(ref_img.repeat(1, 3, 1, 1))

        return torch.mean((pred_img - ref_img) ** 2)


class PerceptualLoss31(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_calculator = list(vgg19(pretrained=True).to('cuda').children())[0][:12] #[:4] #[:18]

        for param in self.loss_calculator.parameters():
            param.requires_grad = False

    def forward(self, pred_img, ref_img):
        self.loss_calculator.eval()
        pred_img = self.loss_calculator(pred_img.repeat(1, 3, 1, 1))
        ref_img = self.loss_calculator(ref_img.repeat(1, 3, 1, 1))

        return torch.mean((pred_img - ref_img) ** 2)


class PerceptualLoss34(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_calculator = list(vgg19(pretrained=True).to('cuda').children())[0][:18] #[:4] #[:12]

        for param in self.loss_calculator.parameters():
            param.requires_grad = False

    def forward(self, pred_img, ref_img):
        self.loss_calculator.eval()
        pred_img = self.loss_calculator(pred_img.repeat(1, 3, 1, 1))
        ref_img = self.loss_calculator(ref_img.repeat(1, 3, 1, 1))

        return torch.mean((pred_img - ref_img) ** 2)

