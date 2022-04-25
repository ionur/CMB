from utils import *
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import RandomCrop, Normalize
from model import *

### Set seeds and device
rand_seed = 19951202
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

if torch.cuda.is_available():
  torch.cuda.manual_seed_all(rand_seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Device: ', device)


### Load data
kappa_qe_test = torch.from_numpy(np.load('../data/K_QE_test.npy'))
kappa_tru_test = torch.from_numpy(np.load('../data/K_test.npy'))
dataset = CMBDataset(kappa_qe_test.unsqueeze(1), kappa_tru_test.unsqueeze(1), Normalize(0,1), Normalize(0,1))
loader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)
window = np.load('../data/window_info.npz')['taper']


loss_net = Autoencoder().to(device)
loss_net.load_state_dict(torch.load('./models/ae_lossnet.pt'))
# gen_net =  UNet2D(in_channels=1, out_channels=1, conv_depths=(8, 16, 32, 64)).to(device)
# gen_net.load_state_dict(torch.load('./models/ae_lossnet.pt'))
loss_net.eval()
loss_net.to(device)

## variables to set up the size of the map
N = int(2**7)  # this is the number of pixels in a linear dimension
pix_size  = 2.34375 # size of a pixel in arcminutes

## variables to set up the map plots
X_width = N*pix_size/60  # horizontal map width in degrees
Y_width = N*pix_size/60  # vertical map width in degrees

f1 = 0
f2 = 0
with torch.no_grad():
    for i, (X, _, Y) in enumerate(loader):
        out = loss_net(Y.to(device))
        out = out.squeeze()
        Y = Y.squeeze()

        Y = Y.numpy()
        out = out.cpu().numpy()

        # Plot predictions
        tru_map = Y
        c_min = -max(-np.min(tru_map), np.max(tru_map))
        c_max = +max(-np.min(tru_map), np.max(tru_map)) 
        p = Plot_CMB_Map(tru_map,c_min,c_max,X_width,Y_width, path='./plots/actual.png')

        pred_map = out * window
        c_min = -max(-np.min(pred_map), np.max(pred_map))  # minimum for color bar
        c_max = +max(-np.min(pred_map), np.max(pred_map))  # maximum for color bar
        p = Plot_CMB_Map(pred_map,c_min,c_max,X_width,Y_width, path='./plots/pred.png')

        diff_map = tru_map - pred_map
        c_min = -max(-np.min(diff_map), np.max(diff_map))  # minimum for color bar
        c_max = +max(-np.min(diff_map), np.max(diff_map))  # maximum for color bar
        p = Plot_CMB_Map(diff_map,c_min,c_max,X_width,Y_width, path='./plots/diff.png')

        # ## make a power spectrum
        binned_ell, binned_spectrum1 = calculate_2d_spectrum(tru_map, tru_map, pix_size,N)
        _, binned_spectrum2 = calculate_2d_spectrum(pred_map, pred_map, pix_size,N)
        # _, binned_spectrum3 = calculate_2d_spectrum(X.cpu().numpy()[0,0], X.cpu().numpy()[0,0], pix_size, N)

        print(r(tru_map, pred_map))
        print('MSE: ', np.sum(np.square(np.abs(tru_map - pred_map)))/(128*128))
        print('Done')
        input()

        f1 += binned_spectrum1
        f2 += binned_spectrum2
        
plt.figure()
# plt.semilogy(binned_ell,binned_spectrum1)
# plt.semilogy(binned_ell,binned_spectrum2)
plt.semilogy(binned_ell,f1/ len(dataset))
plt.semilogy(binned_ell,f2/len(dataset))
# plt.semilogy(binned_ell,binned_spectrum3)
plt.ylabel('$C_{L}$ [$\mu$K$^2$]')
plt.xlabel('$L$')
plt.legend(['True Power Spectrum', 'Reconstructed Power Spectrum'])
plt.savefig('power_spectrum.pdf')
plt.close()
       
  

