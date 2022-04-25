import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Dataset

def r(m1,m2):
    m1 = m1 - np.mean(m1)
    m2 = m2 - np.mean(m2)
    r = np.sum(m1*m2)
    r = r/ np.sqrt(np.sum(m1**2)*np.sum(m1**2))
    return r

def Plot_CMB_Map(Map_to_Plot,c_min,c_max,X_width,Y_width, path):
    print("map mean:",np.mean(Map_to_Plot),"map rms:",np.std(Map_to_Plot))
    plt.figure(figsize=(5,5))
    im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
    im.set_clim(c_min,c_max)
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.25)

    cbar = plt.colorbar(im,fraction=0.047)
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel('angle $[^\circ]$')
    plt.xlabel('angle $[^\circ]$')
    plt.savefig(path)


def calculate_2d_spectrum(Map1,Map2,pix_size,N, delta_ell=50,ell_max=3500):
    "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"
    N=int(N)

    # make a 2d ell coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    
    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array = np.zeros(N_bins)
    
    # get the 2d fourier transform of the map
    FMap1 = np.fft.ifft2(np.fft.fftshift(Map1))
    FMap2 = np.fft.ifft2(np.fft.fftshift(Map2))
    PSMap = np.fft.fftshift(np.real(np.conj(FMap1) * FMap2))
    # fill out the spectra
    i = 0
    while (i < N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array[i] = np.mean(PSMap[inds_in_bin])
        #print i, ell_array[i], inds_in_bin, CL_array[i]
        i = i + 1
 
    # return the power spectrum and ell bins
    return(ell_array,CL_array*np.sqrt(pix_size /60.* np.pi/180.)*2.)


# Define dataset classes
class CMBDatasetPretrain(Dataset):
    
    def __init__(self, imgs, labels, transform):

      self.imgs = imgs 
      self.labels = labels               
      self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
      img = self.imgs[idx]
      label = self.labels[idx]

      if self.transform:
        img = self.transform(img)
          
      return img, label

class CMBDatasetPretrainAE(Dataset):
    
    def __init__(self, imgs, transform=None):

      self.imgs = imgs     
      self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
      img = self.imgs[idx]

      if self.transform:
        img = self.transform(img)
          
      return img


class CMBDataset(Dataset):
    
    def __init__(self, X, Y, train_transform, test_transform):

      self.X = X 
      self.Y = Y               
      self.train_transform = train_transform
      self.test_transform = test_transform


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
      X = self.X[idx]
      Y = self.Y[idx]
      Y_transformed = self.Y[idx]
      if self.train_transform:
        X = self.train_transform(X)
      if self.test_transform:
         Y_transformed = self.test_transform(Y_transformed)
          
      return X, Y_transformed, Y
