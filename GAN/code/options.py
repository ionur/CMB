# ##################################################################### #
# Modifications authored by:
# Peikai Li
# Ipek Ilayda Onur
#
# Most of the code is borrowed from:
# https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms/blob/master/options.py
# ##################################################################### #

import os

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
        self.dataset_dir = 'data_2n'
        self.checkpoint_dir = 'data_2n'
        self.input_ch = 1
        self.batch_size = 1
        self.dataset_name = 'CMB'
        self.data_type = 32
        self.image_mode = 'png'
        self.n_downsample = 4
        self.n_residual = 9
        self.n_workers = 8
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