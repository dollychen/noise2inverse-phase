import numpy as np
from skimage import exposure
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

#util functions
#use skimage to calculate psnr
def psnr_skimage(img1, img2, data_range=65536, cuda=True):
    #covert to cpu
    if cuda:
        img1 = np.squeeze(img1.detach().cpu().numpy())
        img2 = np.squeeze(img2.detach().cpu().numpy())
    return psnr(img1, img2, data_range=data_range)

#use skimage to calculate ssim
def ssim_skimage(img1, img2, data_range=65536, cuda=True):
    #covert to cpu
    if cuda:
        img1 = np.squeeze(img1.detach().cpu().numpy())
        img2 = np.squeeze(img2.detach().cpu().numpy())
    return ssim(img1, img2, data_range=data_range)

#use np percetile to stretch image histogram and crop 5 percent top and bottom
def percetile_stretch(img, top=5, bottom=95):
    img = img.detach().cpu().numpy()
    ptop, pbottom = np.percentile(img, (top, bottom))
    img_rescale = exposure.rescale_intensity(img, in_range=(ptop, pbottom))
    return img_rescale

def network_setup(network, multi_gpu, n_features=32, n_input_channels=1):
    """_summary_

    Args:
        network (string): unet, dncnn, or msd

    """
    # Option a) Use MSD network, not available for windows environment
    if network == "msd":
        from msd_pytorch import MSDRegressionModel
        model = MSDRegressionModel(n_input_channels, 1, 100, 1, parallel=multi_gpu)
        net = model.net
        optim = model.optimizer

    # Option b) Use UNet
    if network == "unet":
        from noise2inverse import UNet
        net = UNet(n_input_channels, 1,n_features=n_features).cuda() # 1 input channel, 1 output channel
        if multi_gpu:
            print("Using multi-gpu")
            net = nn.DataParallel(net)
            #net = nn.parallel.DistributedDataParallel(net)

        optim = torch.optim.Adam(net.parameters())

    # Option c) Use DnCNN
    if network == "dncnn":
        from noise2inverse import DnCNN
        net = DnCNN(1, features=32).cuda() # 1 input channel, 1 output channel
        if multi_gpu:
            print("Using multi-gpu")
            net = nn.DataParallel(net, device_ids=[0,1])

        optim = torch.optim.Adam(net.parameters())
    return net, optim

#get signal to noise ratio of a patch in an image
def snr_patch(img, patch_size=256, location=(1100,350), cuda=True):
    if cuda:
        img = np.squeeze(img.detach().cpu().numpy())
    img_background = img[location[0]:location[0]+patch_size, location[1]:location[1]+patch_size]
    #img signal is in the middle of the image
    (h,w) = img.shape
    img_signal = img[h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2]
    return np.mean(img_signal) / np.std(img_background)



