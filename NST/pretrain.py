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
kappa_qe_train = torch.from_numpy(np.load('../data/K_QE_train.npy'))
kappa_qe_test = torch.from_numpy(np.load('../data/K_QE_test.npy'))
kappa_tru_train = torch.from_numpy(np.load('../data/K_train.npy'))
kappa_tru_test = torch.from_numpy(np.load('../data/K_test.npy'))

train_imgs = torch.cat((kappa_qe_train, kappa_tru_train),dim=0).unsqueeze(1)
train_labels = 1.0*torch.cat((torch.zeros(len(kappa_qe_train)), torch.ones(len(kappa_tru_train))),dim=0)
test_imgs = torch.cat((kappa_qe_test, kappa_tru_test),dim=0).unsqueeze(1)
test_labels = 1.0*torch.cat((torch.zeros(len(kappa_qe_test)), torch.ones(len(kappa_tru_test))),dim=0)
train_ds = CMBDatasetPretrain(train_imgs, train_labels, Normalize(0,1))
test_ds = CMBDatasetPretrain(train_imgs, train_labels, Normalize(0,1))


train_ds, val_ds = random_split(train_ds, [24000, 3000])
train_loader = DataLoader(train_ds, batch_size=128, pin_memory=True, shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=256, pin_memory=True, shuffle=False, num_workers=4)
window = torch.from_numpy(np.load('../data/window_info.npz')['taper']).to(device).float()

### Train model
model = LossNet().to(device)

optimizer = optim.Adam(model.parameters(), lr = 1e-3)
train_losses = []
val_losses = []
train_accs = []
val_accs = []
for epoch in range(3):
    
    train_loss = 0
    train_acc = 0
    model.train()
    for i, (img, label) in enumerate(tqdm(train_loader, position=0, leave=True)):
      img = img.to(device)
      label = label.to(device)
      
      optimizer.zero_grad() 
      pred = model(img).squeeze()
      loss = F.binary_cross_entropy_with_logits(pred, label)
      loss.backward()
      optimizer.step()
      train_loss += loss.detach().cpu()
      train_acc += torch.sum(torch.sigmoid(pred.detach()).round() == label)

    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
      for i, (img, label) in enumerate(tqdm(val_loader, position=0, leave=True)):
        img = img.to(device)
        label = label.to(device)
        pred = model(img).squeeze()
        loss = F.binary_cross_entropy_with_logits(pred, label)
        val_loss += loss.detach().cpu()
        val_acc += torch.sum(torch.sigmoid(pred).round() == label)

    train_losses.append(train_loss.item() / len(train_loader))
    val_losses.append(val_loss.item() / len(val_loader))
    train_accs.append(train_acc.item() / len(train_ds))
    val_accs.append(val_acc.item() / len(val_ds))
    print('Epoch: {0}, Training Acc: {1}, Validation Acc: {2}, '.format(epoch, train_accs[-1], val_accs[-1]))

    if len(val_losses)==1 or (len(val_losses) > 1 and val_losses[-1] <= min(val_losses)):
      torch.save(model.state_dict(), './models/lossnet.pt')

metrics = pd.DataFrame(data={'Train Loss': train_losses, 'Validation Loss' : val_losses, 'Train Acc' : train_accs, 'Validation Acc' : val_accs})
metrics.to_csv('./metrics/lossnet_metrics.csv')