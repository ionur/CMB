# ##################################################################### #
# Modifications authored by:
# Peikai Li
# Ipek Ilayda Onur
#
# Most of the code is borrowed from:
# https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms/blob/master/networks.py
# ##################################################################### #

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import get_grid, get_norm_layer, get_pad_layer, calculate_2d_spectrum
import numpy as np

#import window parameters
window = np.load('../data/window_info.npz')
window = torch.Tensor(window['taper']).to('cuda:0')

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        act = nn.ReLU(inplace=True)
        input_ch = opt.input_ch
        n_gf = opt.n_gf
        norm = get_norm_layer(opt.norm_type)
        output_ch = opt.output_ch
        pad = get_pad_layer(opt.padding_type)

        self.n_downsample = opt.n_downsample
        self.n_residual = opt.n_residual

        model = []
        model.append([nn.Conv2d(input_ch + 1, n_gf, kernel_size=3, padding=1), norm(n_gf), act])

        for _ in range(opt.n_downsample):
            if _ != opt.n_downsample - 1:
                model.append([nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1), norm(2 * n_gf), act])
                n_gf *= 2
            else:
                model.append([nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1, stride=2), norm(2 * n_gf), act])
                n_gf *= 2

        for _ in range(opt.n_residual):
            model.append([ResidualBlock(n_gf, pad, norm, act)])

        for i in range(opt.n_downsample):
            if i == 0:
                model.append([nn.ConvTranspose2d(n_gf, n_gf // 2, kernel_size=3, padding=1, stride=2, output_padding=1),
                              norm(n_gf // 2), act])
            else:
                model.append([nn.Conv2d(n_gf, n_gf // 4, kernel_size=3, padding=1), norm(n_gf // 4), act])
                n_gf //= 2

        model.append([nn.Conv2d(n_gf, output_ch, kernel_size=3, padding=1)])
        self.model = nn.ModuleList()
        for i in range(len(model)):
            self.model.append(nn.Sequential(*model[i]))
        print(self.parameters)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    """
        forward step includes skip connections
    """
    def forward(self, x):
        x_skip = []
        for i in range(self.n_downsample):
            x = self.model[i](x)
            x_skip.append(x)

        x = self.model[self.n_downsample](x)
        for i in range(self.n_residual):
            x = self.model[i + 1 + self.n_downsample](x)
        for i in range(self.n_downsample):
            x = self.model[i + 1 + self.n_residual + self.n_downsample](x)
            x = torch.cat((x, x_skip[self.n_downsample - i - 1]), axis=1)
        return self.model[-1](x) * window


class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = opt.input_ch
        n_df = opt.n_df
        norm = nn.InstanceNorm2d

        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=3, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=3, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=3, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=3, padding=1, stride=1), norm(8 * n_df), act]]

        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        for i in range(opt.n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(opt))
        self.n_D = 2
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
        return result

class Loss(object):
    def __init__(self, opt, modlmap):
        self.opt = opt
        self.modlmap = modlmap
        self.device = torch.device('cuda:0' if opt.gpu_ids != -1 else 'cpu:0')
        self.dtype = torch.float16 if opt.data_type == 16 else torch.float32

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = 2

    def __call__(self, D, G, input, target):
        scale = 10.

        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)

        spectrum_maps, CL_arrays = calculate_2d_spectrum(fake, target)
        PSMap_fake   = spectrum_maps['PSMap_auto1'] * self.modlmap
        Phase_fake   = spectrum_maps['Phase1']
        PSMap_target = spectrum_maps['PSMap_auto2'] * self.modlmap
        Phase_target = spectrum_maps['Phase2']


        loss_PSMap = self.criterion(PSMap_fake, PSMap_target)
        loss_phase = self.criterion(Phase_fake, Phase_target)

        #power spectrum loss is defined as the summation of power spectrum and phase maps with proper scaling
        loss_PS = Variable((loss_PSMap* scale) + (loss_phase/ 20), requires_grad=True)

        loss_L2 = self.criterion(fake, target)
        loss_P  = self.log_power_spectrum_loss(fake, target, scale=scale)

        real_features = D(target)
        fake_features = D(fake.detach())

        #add PSMap loss to the discriminator
        loss_D += loss_PS

        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device, self.dtype)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        fake_features = D(fake)

        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())

            loss_G += loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM

        return loss_D, loss_G, target, fake, loss_L2, loss_P/scale

    def log_power_spectrum_loss(self, fake, target, scale=100):
        batch_size = fake.shape[0]
        fake = torch.squeeze(fake, dim=1)
        target = torch.squeeze(target, dim=1)
        fake_ps = []
        target_ps = []
        for i in range(batch_size):
            fake_ps.append(calculate_2d_spectrum(fake[i])[0])
            target_ps.append(calculate_2d_spectrum(target[i])[0])

        fake_ps = torch.stack(fake_ps)
        target_ps = torch.stack(target_ps)
        log_loss = self.criterion(torch.log(fake_ps), torch.log(target_ps)) * scale
        return log_loss


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)
