# ##################################################################### #
# Authored by:
# Peikai Li
# Ipek Ilayda Onur
#
# Some code is borrowed from:
# https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms/blob/master/train.py
# ##################################################################### #

import os
import torch
from torch.utils.data import DataLoader
from options import BaseOption,TrainOption
from networks import Discriminator, Generator, Loss
from pipeline import CustomDataset
from utils import Manager, update_lr, weights_init, calculate_signal_to_noise, calculate_2d_spectrum, calculate_signal_to_noise
import numpy as np
 from tqdm import tqdm
import datetime
import torch.nn.functional as F
import optuna
import logging
import copy

class TrainCMB(object):
    def __init__(self, opts, modlmap):
        opt, val_opt = opts[0], opts[1]
        self.device = torch.device('cuda:0')
        self.dtype = torch.float16 if opt.data_type == 16 else torch.float32

        train_dataset = CustomDataset(opt)
        val_dataset   = CustomDataset(val_opt)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.n_workers,
                                       shuffle=not opt.no_shuffle)

        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=val_opt.batch_size,
                                     num_workers=val_opt.n_workers,
                                     shuffle=not val_opt.no_shuffle)

        self.G = Generator(opt).apply(weights_init).to(device=self.device, dtype=self.dtype)
        self.D = Discriminator(opt).apply(weights_init).to(device=self.device, dtype=self.dtype)

        self.criterion = Loss(opt, modlmap)

        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

        self.val_loss = []

        # parameters for power spectrum loss
        ## variables to set up the size of the map
        self.num_pix = int(2 ** 7)  # this is the number of pixels in a linear dimension
        self.pix_size = 2.34375  # size of a pixel in arcminutes
        self.delta_ell = 50.
        self.ell_max = 3500.

    def __call__(self, opt):
        if opt.latest and os.path.isfile(opt.model_dir + '/' + str(opt.latest) + '_dict.pt'):
            pt_file = torch.load(opt.model_dir + '/' + str(opt.latest) + '_dict.pt')
            init_epoch = pt_file['Epoch']
            print("Resume at epoch: ", init_epoch)
            self.G.load_state_dict(pt_file['G_state_dict'])
            self.D.load_state_dict(pt_file['D_state_dict'])
            self.G_optim.load_state_dict(pt_file['G_optim_state_dict'])
            self.D_optim.load_state_dict(pt_file['D_optim_state_dict'])
            current_step = init_epoch * len(self.data_loader)

            for param_group in self.G_optim.param_groups:
                lr = param_group['lr']
        else:
            init_epoch = 1
            current_step = 0

        manager = Manager(opt)

        total_step = opt.n_epochs * len(self.train_loader)
        start_time = datetime.datetime.now()

        init_lr = opt.lr
        lr = opt.lr

        for epoch in range(init_epoch, opt.n_epochs + 1):
            for input, target in tqdm(self.train_loader):
                self.G.train()

                current_step += 1
                input, target = input.to(device=self.device, dtype=self.dtype), target.to(self.device, dtype=self.dtype)

                D_loss, G_loss, target_tensor, generated_tensor, L2_loss, C_loss = self.criterion(self.D, self.G, input,
                                                                                                  target)

                # torch.autograd.set_detect_anomaly(True)
                self.G_optim.zero_grad()
                G_loss.backward()
                self.G_optim.step()

                self.D_optim.zero_grad()
                D_loss.backward()
                self.D_optim.step()

                package = {'Epoch': epoch,
                           'current_step': current_step,
                           'total_step': total_step,
                           'D_loss': D_loss.detach().item(),
                           'C_loss': C_loss.detach().item(),
                           'G_loss': G_loss.detach().item(),
                           'L2_loss': L2_loss.detach().item(),
                           'D_state_dict': self.D.state_dict(),
                           'G_state_dict': self.G.state_dict(),
                           'D_optim_state_dict': self.D_optim.state_dict(),
                           'G_optim_state_dict': self.G_optim.state_dict(),
                           'target_tensor': target_tensor,
                           'generated_tensor': generated_tensor.detach()
                           }

                manager(package)

                if opt.val_during_train:
                    self.G.eval()
                    for p in self.G.parameters():
                        p.requires_grad_(False)

                    val_loss_ll = 0
                    val_loss_ps = 0
                    #R coefficient
                    R_loss = 0
                    #signal to noise
                    snr = []
                    for input, target in tqdm(self.val_loader):
                        input, target = input.to(device=self.device, dtype=self.dtype), target.to(self.device,
                                                                                                  dtype=self.dtype)
                        fake = self.G(input)
                        ll = F.mse_loss(fake, target)
                        val_loss_ll += ll.cpu().detach()

                        #calculate power spectrums and phase maps
                        power_spectrums = torch.tensor([calculate_2d_spectrum(f.squeeze().cpu().detach(), t.squeeze().cpu().detach()) for f, t
                                            in zip(fake, target)])

                        PSMap_true      = torch.tensor([ s[1]['CL_auto1'] * np.sqrt(self.pix_size / 60. * np.pi / 180.) * 2. for s in power_spectrums])
                        PSMap_fake      = torch.tensor([ s[1]['CL_auto2'] * np.sqrt(self.pix_size / 60. * np.pi / 180.) * 2.for s in power_spectrums])

                        val_loss_ps += F.mse_loss(torch.log(PSMap_true[:, 1:]), torch.log(PSMap_fake[:, 1:])).cpu().detach()

                        snr.append(torch.tensor([calculate_signal_to_noise(s[1]) for s in power_spectrums]))

                        # R metric is fake*target/sqrt(fake^2 target^2)
                        target_flat = target.reshape([opt.batch_size, -1])
                        fake_flat = fake.reshape([opt.batch_size, -1])

                        target_sq = torch.sum(torch.mul(target_flat, target_flat), dim=1)
                        fake_sq = torch.sum(torch.mul(fake_flat, fake_flat), dim=1)
                        R = torch.sum(torch.mul(fake_flat, target_flat), dim=1) / torch.sqrt(
                            torch.mul(target_sq, fake_sq))
                        R_loss -= (torch.sum(R.cpu().detach()) / opt.batch_size)
                    snr = torch.mean(torch.cat(snr, dim=0), dim=0)
                    self.val_loss.append([val_loss_ll / len(self.val_loader), val_loss_ps / len(self.val_loader),
                                          R_loss / len(self.val_loader), snr])
                    for p in self.G.parameters():
                        p.requires_grad_(True)
            if epoch > opt.epoch_decay:
                lr = update_lr(lr, init_lr, opt.n_epochs - opt.epoch_decay, self.D_optim, self.G_optim)

        print("Total time taken: ", datetime.datetime.now() - start_time)


# put any args to search in here
def exp_opts(trial, opt):
    #########
    ### put any param to search here
    ##########

    opt.lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    opt.n_downsample = trial.suggest_int('n_downsample', 2, 3)
    opt.n_residual = trial.suggest_int('n_residual', 2, 8)
    opt.beta1 = trial.suggest_loguniform('beta1', 0.5, 0.9)
    opt.lambda_FM = trial.suggest_int('lambda_FM', 8, 15)

    #############
    #### also make your additions here when you add opts !!
    ##############
    out_str = 'MIN VAL ERROR n_downsample{}_n_residual{}_lr{}_beta1{}_lambda_FM{} IS '.format(opt.n_downsample,
                                                                                              opt.n_residual, opt.lr,
                                                                                              opt.beta1, opt.lambda_FM)

    opt.dataset_name = 'n_downsample{}_n_residual{}_lr{}_beta1{}_lambda_FM{} IS '.format(opt.n_downsample,
                                                                                         opt.n_residual, opt.lr,
                                                                                         opt.beta1, opt.lambda_FM)
    return opt, out_str


def objective(trial, opts):
    # Get the hparams suggested by optuna
    opt, out_str = exp_opts(trial, opts)

    val_opt = copy.copy(opt)
    val_opt.is_train = False
    val_opt.is_val = True  ####
    val_opt.no_shuffle = True

    modlmap = np.load('../data/modlmap.npy')
    modlmap = torch.Tensor(modlmap).to('cuda:0')

    model = TrainCMB([opt, val_opt], modlmap)
    model(opt)

    # [ll loss, power spectrum loss, R_loss, SnR loss]
    val_loss = torch.tensor(model.val_loss).reshape((-1, 4))

    ll = val_loss[:, 0]
    ps = val_loss[:, 1]
    R = val_loss[:, 2]
    ps_ = (10 ** 10) * ps
    snr = val_loss[:, 3]


    # max SnR
    min_idx = torch.argmin(snr[4:])
    min_val_loss = torch.min(snr[4:])

    print(val_loss)
    print('{} {} . LL loss is {} and PS loss is {}'.format(out_str, min_val_loss, ll[min_idx], ps[min_idx]))
    return min_val_loss

if __name__ == '__main__':
    base_opt = BaseOption()
    opt = TrainOption()
    opt.dataset_name = 'CMB'
    opt.input_ch = 1
    opt.n_epochs = 12
    opt.norm_type = 'InstanceNorm2d'
    opt.batch_size = 32
    opt.n_gf = 32
    opt.report_freq = 100
    opt.save_freq = 282000000000000
    opt.display_freq = 10000000000
    opt.val_during_train = True

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("../studies/data_1n/hyperparam_search.log", mode="w"))
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    val_opt = copy.copy(opt)
    val_opt.is_train = False
    val_opt.is_val = True  ####
    val_opt.no_shuffle = True

    opt.n_workers = 8

    opt.n_epochs = 30
    opt.lr = 0.0003
    opt.n_downsample = 3
    opt.n_residual = 5
    opt.beta1 = 0.5
    opt.lambda_FM = 10


    # If a study already exists, load that one
    study_name = 'study'
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        # storage='sqlite:///content/drive/My Drive/Colab Notebooks/10707/studies/{}.db'.format(study_name),
        load_if_exists=False  # If file exists, load it and resume instead
    )

    ###### put how many trials here!!
    logger.info("Start optimization.")
    study.optimize(lambda trial: objective(trial, opt), n_trials=10)

    with open("../studies/data_1n/hyperparam_search.log") as f:
        assert f.readline() == "Start optimization.\n"
        assert f.readline().startswith("Finished trial#0 with value:")