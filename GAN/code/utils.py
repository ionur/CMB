# ##################################################################### #
# Authored by:
# Peikai Li
# Ipek Ilayda Onur
# ##################################################################### #

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import os
from functools import partial
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# index for power spectrum
ind_ps = np.load('../data/ind_ps.npy', allow_pickle=True).tolist()

"""
Plots CMB maps
"""


def Plot_CMB_Map(Map_to_Plot, c_min, c_max, X_width, Y_width):
    print("map mean:", np.mean(Map_to_Plot), "map rms:", np.std(Map_to_Plot))
    plt.figure(figsize=(10, 10))
    im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower', cmap=cm.RdBu_r)
    im.set_clim(c_min, c_max)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)

    cbar = plt.colorbar(im, cax=cax)
    im.set_extent([0, X_width, 0, Y_width])
    plt.ylabel('angle $[^\circ]$')
    plt.xlabel('angle $[^\circ]$')
    plt.show()
    return (0)


"""
    Calculates 2D power spectrum using Fast Fourier Transform

    input: Map1
           Map2 

    output: 
        CL_array
        FMap  : Fourier transform map
        PSMap : power spectrum map
        Phase : Phase

"""


def calculate_2d_spectrum(Map1, Map2):
    # get the 2d fourier transform of the maps
    FMap1 = torch.fft.ifft2(np.fft.fftshift(Map1))
    FMap2 = torch.fft.ifft2(np.fft.fftshift(Map2))
    PSMap_cross = torch.fft.fftshift(np.real(np.conj(FMap1) * FMap2))
    PSMap_auto1 = torch.fft.fftshift(np.real(np.conj(FMap1) * FMap1))
    PSMap_auto2 = torch.fft.fftshift(np.real(np.conj(FMap2) * FMap2))
    Phase1 = torch.angle(FMap1)
    Phase2 = torch.angle(FMap2)

    maps = {
        'FMap1': FMap1,
        'FMap2': FMap2,
        'PSMap_cross': PSMap_cross,
        'PSMap_auto1': PSMap_auto1,
        'PSMap_auto2': PSMap_auto2,
        'Phase1': Phase1,
        'Phase2': Phase2
    }

    N_bins = 69
    ell_array = torch.arange(N_bins)
    CL_cross = torch.zeros(N_bins)
    CL_auto1 = torch.zeros(N_bins)
    CL_auto2 = torch.zeros(N_bins)
    n_l = torch.zeros(N_bins)

    for i in range(N_bins):
        CL_cross[i] = torch.mean(PSMap_cross[ind_ps[i]])
        CL_auto1[i] = torch.mean(PSMap_auto1[ind_ps[i]])
        CL_auto2[i] = torch.mean(PSMap_auto2[ind_ps[i]])
        n_l[i] = len(ind_ps[i])

    CL_arrays = {
        'CL_cross': CL_cross,
        'CL_auto1': CL_auto1,
        'CL_auto2': CL_auto2,
        'n_l': n_l
    }
    return maps, CL_arrays


"""
Calculates the signal to noise ratio of Map1 and Map2
where CL arrays of the maps are given in the CL_array object
"""


def calculate_signal_to_noise(CL_arrays):
    CL_cross = CL_arrays['CL_cross']
    CL_auto1 = CL_arrays['CL_auto1']
    CL_auto2 = CL_arrays['CL_auto2']
    n_l = CL_arrays['n_l']

    # return the power spectrum and ell bins
    return CL_cross ** 2 * n_l / 2 / CL_auto1 / CL_auto2


# ##################################################################### #
# Taken from
# https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms/blob/master/utils.py
# ##################################################################### #

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid


def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer


def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer


class Manager(object):
    def __init__(self, opt):
        self.opt = opt
        self.dtype = opt.data_type

    @staticmethod
    def report_loss(package):
        print(
            "Epoch: {} [{:.{prec}}%] Current_step: {} D_loss: {:.{prec}}  G_loss: {:.{prec}}  L2_loss: {:.{prec}} C2_loss: {:.{prec}}".
                format(package['Epoch'], package['current_step'] / package['total_step'] * 100, package['current_step'],
                       package['D_loss'], package['G_loss'], package['L2_loss'], package['C_loss'], prec=5))

    def adjust_dynamic_range(self, data, drange_in, drange_out):
        if drange_in != drange_out:
            if self.dtype == 32:
                scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                        np.float32(drange_in[1]) - np.float32(drange_in[0]))
                bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            elif self.dtype == 16:
                scale = (np.float16(drange_out[1]) - np.float16(drange_out[0])) / (
                        np.float16(drange_in[1]) - np.float16(drange_in[0]))
                bias = (np.float16(drange_out[0]) - np.float16(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def tensor2image(self, image_tensor):
        np_image = image_tensor.squeeze().cpu().float().numpy()
        # if len(np_image.shape) == 3:
        #     np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        # else:
        #     pass

        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def save_image(self, image_tensor, path):
        Image.fromarray(self.tensor2image(image_tensor)).save(path, self.opt.image_mode)

    def save(self, package, image=False, model=False):
        if image:
            path_real = os.path.join(self.opt.image_dir, str(package['Epoch']) + '_' + 'real.png')
            path_fake = os.path.join(self.opt.image_dir, str(package['Epoch']) + '_' + 'fake.png')
            self.save_image(package['target_tensor'], path_real)
            self.save_image(package['generated_tensor'], path_fake)

        elif model:
            path_D = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'D_N2_fv.pt')
            path_G = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'G_N2_fv.pt')
            torch.save(package['D_state_dict'], path_D)
            torch.save(package['G_state_dict'], path_G)

    def __call__(self, package):
        if package['current_step'] % self.opt.display_freq == 0:
            self.save(package, image=True)

        if package['current_step'] % self.opt.report_freq == 0:
            self.report_loss(package)

        if package['save_alert'] < 0.057:
            self.save(package, model=True)


def update_lr(old_lr, init_lr, n_epoch_decay, D_optim, G_optim):
    delta_lr = init_lr / n_epoch_decay
    new_lr = old_lr - delta_lr

    for param_group in D_optim.param_groups:
        param_group['lr'] = new_lr

    for param_group in G_optim.param_groups:
        param_group['lr'] = new_lr

    print("Learning rate has been updated from {} to {}.".format(old_lr, new_lr))

    return new_lr


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)