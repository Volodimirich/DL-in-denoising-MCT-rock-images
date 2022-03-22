from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
import torch.optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models import skip

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def check_int_size(intgr):
    intgr = int(intgr)
    if 0 < intgr <= 400:
        return intgr
    else:
        raise ValueError("Incorrect image crop size, should be integer 0 < X <= 400")


def create_denoised_dir(path_noised):
    path_denoised = path_noised.rstrip("\/") + "_denoised"
    if not os.path.exists(path_denoised):
        os.mkdir(path_denoised)
    return path_denoised


def crop_image(img, size):
    if size > img.size[0]:
        new_size = img.size
    else:
        new_size = (size, size)
    bbox = [int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2)]
    img_cropped = img.crop(bbox)
    print()
    return img_cropped


def get_noise(input_depth, spatial_size, var=1./10):
    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)
    net_input.normal_()  # fill noise
    net_input *= var
    return net_input


def load_cropped_image(file_name, size):
    try:
        img_pil = Image.open(file_name)
    except Exception as e:
        print(f"Failed to open the image {e}")
        assert False
    cropped_img_pil = crop_image(img_pil, size)
    cropped_img_np = np.array(cropped_img_pil).astype(np.float32) / 255
    if len(cropped_img_np.shape) == 3:
        cropped_img_np = np.transpose(cropped_img_np, (2, 0, 1))
    else:
        cropped_img_np = cropped_img_np[None, ...]
    return cropped_img_pil, cropped_img_np


def get_param(net):
    params = []
    params += [x for x in net.parameters()]
    return params


def process_dip(img_noisy_pil, img_noisy_np, path_denoised_for_saving_metrics=None):
    # SETUP
    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
    pad = 'reflection'
    input_depth = 32
    skip_n33d = 128
    skip_n33u = 128
    skip_n11 = 4
    num_scales = 5

    net = skip(num_input_channels=input_depth,
               num_output_channels=1,
               num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d,
               num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u,
               num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11,
               upsample_mode='bilinear',
               downsample_mode='stride',
               need_sigmoid=True,
               need_bias=True,
               pad=pad,
               act_fun='LeakyReLU').type(dtype)

    net_input = get_noise(input_depth, (img_noisy_pil.size[1], img_noisy_pil.size[0])).type(dtype).detach()  # ok
    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()]);  # ok
    print('Number of params: %d' % s)  # ok

    list_psnr = []
    list_loss = []

    # Loss
    mse = torch.nn.MSELoss().type(dtype)  # ok
    img_noisy_torch = torch.from_numpy(img_noisy_np)[None, :].type(dtype)  # ok

    # OPTIMIZE
    num_iter = 700
    lr = 0.000015
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    p = get_param(net)

    optimizer = torch.optim.Adam(p, lr=lr)
    for j in range(num_iter):
        optimizer.zero_grad()

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])

        if j % 100 == 0:
            print(f'Iteration {j}   Loss: {total_loss.item()}   PSNR_noisy: {psrn_noisy}')
        list_loss.append(total_loss.item())
        list_psnr.append(psrn_noisy)

        optimizer.step()

    if path_denoised_for_saving_metrics is not None:
        # save metrics
        pth = path_denoised_for_saving_metrics[:-4]
        pth_psnr = pth + "_psnr.txt"
        pth_loss = pth + "_loss.txt"
        with open(pth_psnr, 'w') as f:
            for item in list_psnr:
                f.write(f"{item}\n")
        with open(pth_loss, 'w') as f:
            for item in list_loss:
                f.write(f"{item}\n")

    return net(net_input).detach().cpu().numpy()[0]


def save_img(path_dir, img):
    img *= 255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im.save(path_dir)


def denoise(path_orig, path_denoised, crop_size):
    for file in os.listdir(path_orig):
        if file.endswith(".jpg"):
            file_path_orig = path_orig + "/" + file
            file_path_den = path_denoised + "/" + file
            img_noisy_pil, img_noisy_np = load_cropped_image(file_path_orig, size=crop_size)
            denoised_img = process_dip(img_noisy_pil, img_noisy_np)
            save_img(file_path_den, denoised_img[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('path_from', type=dir_path)
    parser.add_argument('crop_size', type=check_int_size, default=128)
    args = parser.parse_args()

    denoise(path_orig=args.path_from,
            path_denoised=create_denoised_dir(args.path_from),
            crop_size=args.crop_size)
