import sys
sys.path.append('../')

import optuna
import os
import numpy as np
import torch
import logging
import copy
from train import TrainCMB


class BaseOption(object):
    def __init__(self):
        self.is_train = True
        self.n_epochs = 150
        self.latest = 0
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.epoch_decay = 100
        self.GAN_type = 'LSGAN'
        self.lambda_FM = 10
        self.lr = 0.0002
        self.n_D = 2   
        self.no_shuffle = False
        self.gpu_ids = 0
        self.data_format_input = 'npy'
        self.data_format_target = 'npy'
        self.dataset_dir = '/content/drive/My Drive/Colab Notebooks/10707/Data/debugging'
        self.checkpoint_dir = '/content/drive/My Drive/Colab Notebooks/10707/checkpoint'
        self.input_ch = 1
        self.saturation_lower_limit_input = 1
        self.saturation_upper_limit_input = 200
        self.saturation_lower_limit_target = -3000
        self.saturation_upper_limit_target = 3000
        self.batch_size = 1
        self.dataset_name = 'CMB'
        self.data_type = 32
        self.image_mode = 'png'
        self.n_downsample = 4
        self.n_residual = 9
        self.n_workers = 1
        self.norm_type = 'InstanceNorm2d'
        self.padding_type = 'zero'
        self.padding_size = 0
        self.max_rotation_angle = 0
        self.val_during_train = False
        self.report_freq = 10
        self.save_freq = 10000
        self.display_freq = 100
        self.format = 'png'
        self.n_df = 64
        self.flip = 64
        self.n_gf = 32
        self.output_ch = 1

        if self.data_type == 16:
            self.eps = 1e-4
        elif self.data_type == 32:
            self.eps = 1e-8

        dataset_name = self.dataset_name
        checkpoint_dir = self.checkpoint_dir
        os.makedirs(os.path.join(checkpoint_dir, dataset_name, 'Image', 'Train'), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, dataset_name, 'Image', 'Test'), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, dataset_name, 'Model'), exist_ok=True)

        if self.is_train:
            self.image_dir = os.path.join(checkpoint_dir, dataset_name, 'Image/Train')
        else:
            self.image_dir = os.path.join(checkpoint_dir, dataset_name, 'Image/Test')
        self.model_dir = os.path.join(checkpoint_dir, dataset_name, 'Model')

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()
        self.is_train = True
        self.n_epochs = 150
        self.latest = 0
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.epoch_decay = 100
        self.GAN_type = 'LSGAN'
        self.lambda_FM = 10
        self.lr = 0.0002
        self.n_D = 2   
        self.no_shuffle = False

base_opt = BaseOption()
opt = TrainOption()
opt.dataset_name = 'CMB' 
opt.input_ch =  1 
opt.n_downsample = 3
opt.n_residual = 5
opt.n_epochs =  5
opt.norm_type = 'InstanceNorm2d'
opt.batch_size =  128
opt.report_freq =  1
opt.beta1 = 0.5
opt.save_freq =  281
opt.lr = 0.0002
opt.display_freq  = 10000000000
opt.val_during_train = True
opt.dataset_dir = '../Data/debugging'
opt.checkpoint_dir = '../checkpoint'


logging_fname = "../studies/hyperparam_search.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(logging_fname, mode="w"))
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.



#put any args to search in here
def exp_opts(trial, opt):


    #########
    ### put any param to search here
    ##########
    opt.lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    opt.n_downsample = trial.suggest_int('n_downsample', 2, 4)
    opt.n_residual = trial.suggest_int('n_residual', 2, 8)

    #############
    #### also make your additions here when you add opts !!
    ##############
    out_str = 'MIN VAL ERROR n_downsample{}_n_residual{}_lr{}_ IS '.format(opt.n_downsample,opt.n_residual,opt.lr)
    return opt, out_str

def objective(trial, opts):
    # Get the hparams suggested by optuna
    opt, out_str = exp_opts(trial, opts)

    val_opt = copy.copy(opt)
    val_opt.is_train = False
    val_opt.is_val = True
    val_opt.no_shuffle = True

    model = TrainCMB([opt, val_opt])
    model(opt)
    
    #first col indicates ll loss, second col indicates power spectrum loss
    val_loss = torch.tensor(model.val_loss).reshape((-1,2))

    ll     = val_loss[:,0]
    ps     = val_loss[:,1]
    ps_    = (10**7) * ps
    
    min_idx = torch.argmin(ps_+ll)
    min_val_loss = torch.min(ps_+ll)
    
    print('{} {} . LL loss is {} and PS loss is {}'.format(out_str, min_val_loss, ll[min_idx], ps[min_idx]))
    return min_val_loss

# If a study already exists, load that one
study_name = 'study'
study = optuna.create_study(
        direction='minimize', 
        study_name=study_name,
        # storage='sqlite:///content/drive/My Drive/Colab Notebooks/10707/studies/{}.db'.format(study_name),
        load_if_exists=True # If file exists, load it and resume instead
    )


###### put how many trials here!!
logger.info("Start optimization.")
study.optimize(lambda trial: objective(trial, opt), n_trials=2)

with open(logging_fname) as f:
    assert f.readline() == "Start optimization.\n"
    assert f.readline().startswith("Finished trial#0 with value:")