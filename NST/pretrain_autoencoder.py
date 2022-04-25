from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import *
from torchvision.transforms import GaussianBlur, Normalize
import torchvision.transforms as transforms
with torch.no_grad():
    torch.cuda.empty_cache()
from model import *
import pandas as pd

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


### Load data ###
kappa_tru_train = torch.from_numpy(np.load('../data/K_train.npy'))
kappa_tru_test = torch.from_numpy(np.load('../data/K_test.npy'))
train_ds = CMBDatasetPretrainAE(kappa_tru_train.unsqueeze(1))
test_ds = CMBDatasetPretrainAE(kappa_tru_test.unsqueeze(1))

train_ds, val_ds = random_split(train_ds, [13000, 500])
train_loader = DataLoader(train_ds, batch_size=128, pin_memory=True, shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=256, pin_memory=True, shuffle=False, num_workers=4)
window = torch.from_numpy(np.load('../data/window_info.npz')['taper']).to(device).float()

### Train model
# model = UNet2D(in_channels=1, out_channels=1, conv_depths=(8, 16, 32, 64)).to(device)
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80, 90, 95], gamma=0.1)
train_losses = []
val_losses = []

for epoch in range(100):
    
    train_loss = 0
    model.train()
    for i, img in enumerate(tqdm(train_loader, position=0, leave=True)):
      img = img.to(device)
     
      optimizer.zero_grad() 
      pred = model(img)

      loss = F.mse_loss(pred*window, img)
      loss.backward()
      optimizer.step()
      train_loss += loss.detach().cpu()

    scheduler.step()
    val_loss = 0
    model.eval()
    with torch.no_grad():
      for i, img in enumerate(tqdm(val_loader, position=0, leave=True)):
        img = img.to(device)
        pred = model(img)
        loss = F.mse_loss(pred*window, img)
        val_loss += loss.detach().cpu()

    train_losses.append(train_loss.item() / len(train_loader))
    val_losses.append(val_loss.item() / len(val_loader))
    print('Epoch: {0}, Training Loss: {1}, Validation Loss: {2}, '.format(epoch, train_losses[-1], val_losses[-1]))

    if len(val_losses)==1 or (len(val_losses) > 1 and val_losses[-1] <= min(val_losses)):
      torch.save(model.state_dict(), './models/ae_lossnet.pt')

metrics = pd.DataFrame(data={'Train Loss': train_losses, 'Validation Loss' : val_losses})
metrics.to_csv('./metrics/ae_lossnet_metrics.csv')