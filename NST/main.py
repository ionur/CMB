from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import *
from torchvision.transforms import GaussianBlur, Normalize
import torchvision.transforms as transforms
from model import *
import pandas as pd
with torch.no_grad():
    torch.cuda.empty_cache()


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
kappa_qe_train = torch.from_numpy(np.load('../data/K_QE_train.npy'))
kappa_qe_test = torch.from_numpy(np.load('../data/K_QE_test.npy'))
kappa_tru_train = torch.from_numpy(np.load('../data/K_train.npy'))
kappa_tru_test = torch.from_numpy(np.load('../data/K_test.npy'))



dataset = CMBDataset(kappa_qe_train.unsqueeze(1), kappa_tru_train.unsqueeze(1), Normalize(0,1), Normalize(0,1))
train_ds, val_ds = random_split(dataset, [13000, 500])
train_loader = DataLoader(train_ds, batch_size=64, pin_memory=True, shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=256, pin_memory=True, shuffle=False, num_workers=4)
window = torch.from_numpy(np.load('../data/window_info.npz')['taper']).to(device).float()

### Train model
gen_net = UNet2D(in_channels=1, out_channels=1, conv_depths=(8, 16, 32, 64, 128))
gen_net.to(device)


loss_net = Autoencoder().to(device)
loss_net.load_state_dict(torch.load('./models/ae_lossnet.pt'))
loss_net.to(device)
loss_net.eval()


optimizer = optim.Adam(gen_net.parameters(), lr = 1e-3)
train_losses = []
val_losses = []

for epoch in range(100):
    
    train_loss = 0
    gen_net.train()
    for i, (X, Y_transformed, Y) in enumerate(tqdm(train_loader, position=0, leave=True)):
      X = X.to(device)
      Y_transformed = Y.to(device)
      Y = Y.to(device)
      
      optimizer.zero_grad()  # zero the gradient buffers
      Y_hat  = gen_net(X)
      # Y_hat = (Y_hat - torch.mean(Y_hat,dim=[1,2,3],keepdim=True)) #/torch.std(Y_hat, dim=[1,2,3],keepdim=True)
      # smooth_out = smooth_out - torch.mean(Y_hat,dim=[1,2,3],keepdim=True)
      Y_hat = Y_hat * window
      # smooth_out = smooth_out * window

      _, Z_x = loss_net(Y_hat, return_all=True)
      _, Z_y = loss_net(Y, return_all=True)
      
      # Compute style and content losses
      style_loss = 0
      content_loss = 0
      for i in range(len(Z_x)):
        gram_x = torch.flatten(Z_x[i], start_dim=2)
        gram_x = gram_x @ gram_x.transpose(2,1) / (gram_x.shape[1]*gram_x.shape[2])
        gram_y = torch.flatten(Z_y[i], start_dim=2)
        gram_y = gram_y @ gram_y.transpose(2,1) / (gram_y.shape[1]*gram_y.shape[2])
        style_loss += F.mse_loss(gram_x, gram_y)

        # content_loss+=F.mse_loss(Z_x[i],Z_y[i]) / (Z_x[i].shape[1]*Z_x[i].shape[2])

      loss = style_loss + F.mse_loss(Y, Y_hat) #+ 1e-1*F.mse_loss(torch.fft.rfft2(Y_hat).abs(), torch.fft.rfft2(Y).abs())
      loss.backward()
      optimizer.step()
      train_loss += loss.detach().cpu()

    val_loss = 0
    gen_net.eval()
    with torch.no_grad():
      for i, (X, Y_transformed, Y) in enumerate(tqdm(val_loader, position=0, leave=True)):
        X = X.to(device)
        Y = Y.to(device)
        Y_pred = gen_net(X)
        val_loss +=  F.mse_loss(Y_pred * window, Y)

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    print('Epoch: {0}, Training Loss: {1}, Validation Loss: {2}'.format(epoch, train_losses[-1], val_losses[-1]))

    if len(val_losses)==1 or (len(val_losses) > 1 and val_losses[-1] <= min(val_losses)):
      torch.save(gen_net.state_dict(), './models/gennet2.pt')

metrics = pd.DataFrame(data={'Train Loss': train_losses, 'Validation Loss' : val_losses})
metrics.to_csv('./metrics/gennet_metrics2.csv')
