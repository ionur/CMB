import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../')
from networks import Discriminator, Generator, Loss
from pipeline import CustomDataset
from utils import Manager, update_lr, weights_init, calculate_2d_spectrum
import numpy as np
from tqdm import tqdm
import datetime

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TrainCMB(object):
    def __init__(self, opts):
        opt, val_opt = opts[0], opts[1]
        self.device = torch.device('cuda:0' )
        # device = torch.device('cpu')
        self.dtype = torch.float16 if opt.data_type == 16 else torch.float32

        train_dataset = CustomDataset(opt)
        val_dataset  = CustomDataset(val_opt)


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

        self.criterion = Loss(opt)

        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

        self.val_loss = []


        #parameters for power spectrum loss
        ## variables to set up the size of the map
        self.num_pix = int(2**7)  # this is the number of pixels in a linear dimension
        self.pix_size  = 2.34375 # size of a pixel in arcminutes
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
            for input, target, _, _ in tqdm(self.train_loader):
                self.G.train()
                
                current_step += 1
                input, target = input.to(device=self.device, dtype=self.dtype), target.to(self.device, dtype=self.dtype)

                D_loss, G_loss, target_tensor, generated_tensor, L2_loss = self.criterion(self.D, self.G, input, target)

                self.G_optim.zero_grad()
                G_loss.backward()
                self.G_optim.step()

                self.D_optim.zero_grad()
                D_loss.backward()
                self.D_optim.step()

                package = {   'Epoch': epoch,
                              'current_step': current_step,
                              'total_step': total_step,
                              'D_loss': D_loss.detach().item(),
                              'G_loss': G_loss.detach().item(),
                              'L2_loss': L2_loss.detach().item(),
                              'D_state_dict': self.D.state_dict(),
                              'G_state_dict': self.G.state_dict(),
                              'D_optim_state_dict': self.D_optim.state_dict(),
                              'G_optim_state_dict': self.G_optim.state_dict(),
                              'target_tensor': target_tensor,
                              'generated_tensor': generated_tensor.detach()}

                manager(package)
                # if opt.val_during_train:
                if opt.val_during_train and (current_step % opt.save_freq == 0):
                    self.G.eval()
                    # test_image_dir = os.path.join(test_opt.image_dir, str(current_step))
                    # os.makedirs(test_image_dir, exist_ok=True)
                    # test_model_dir = test_opt.model_dir
                    for p in self.G.parameters():
                        p.requires_grad_(False)

                    val_loss_ll = 0
                    val_loss_ps = 0
                    for input, target, _, name in tqdm(self.val_loader):
                        input, target = input.to(device=self.device, dtype=self.dtype), target.to(self.device, dtype=self.dtype)
                        fake = self.G(input)
                        ll = F.mse_loss(fake, target)
                        
                        ####only look at the first batch
                        if ll != 0:
                            val_loss_ll += ll.cpu().detach()
                            # ## make a power spectrum
                            #spectrum ground truth 
                            ps_true = torch.tensor([ calculate_2d_spectrum(t.squeeze().cpu().detach(), t.squeeze().cpu().detach(),self.delta_ell,self.ell_max,self.pix_size,self.num_pix)[1] for t in target])
                            #spectrum generated 
                            ps_fake = torch.tensor([ calculate_2d_spectrum(f.squeeze().cpu().detach(), f.squeeze().cpu().detach(),self.delta_ell,self.ell_max,self.pix_size,self.num_pix)[1] for f in fake])

                            val_loss_ps += torch.tensor([ F.mse_loss(ps_true[i][1:], ps_fake[i][1:]) for i in range(len(ps_true))]).sum().detach()

                            self.val_loss.append([val_loss_ll / len(input), val_loss_ps / len(input)])
                            break
                        # UpIB = opt.saturation_upper_limit_target
                        # LoIB = opt.saturation_lower_limit_target
                            
                        # np_fake = fake.cpu().numpy().squeeze()
                        # np_real = target.cpu().numpy().squeeze()
                        #
                        # manager.save_image(np_fake, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_fake.png'))
                        # manager.save_image(np_real, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_real.png'))

                    for p in self.G.parameters():
                        p.requires_grad_(True)

            if epoch > opt.epoch_decay :
                lr = update_lr(lr, init_lr, opt.n_epochs - opt.epoch_decay, self.D_optim, self.G_optim)

        print("Total time taken: ", datetime.datetime.now() - start_time)
