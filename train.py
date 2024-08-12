from Dataloader import Train_DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from SSIM import SSIM
from tqdm import tqdm
from MAE import MAE_ViT
import random
import gc
import math
from losses import MMDLoss
from torchvision.utils import save_image
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
datasets_dir = './SAM'
model_exit = './model'
os.makedirs('./model', exist_ok=True)
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = MAE_ViT()
model_name = 'AE-SAM'
base_lr = 0.0001
epochs = 300

if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)

Dataset = Train_DataLoader(datasets_dir)
Datasets = Dataset.datasets_names
traindataloader = torch.utils.data.DataLoader(Dataset, batch_size=64, shuffle=True, num_workers=8)

max_iterations = len(traindataloader) * epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
criterion = SSIM()
cri = nn.CrossEntropyLoss()
mmd = MMDLoss(3)
iterations = 0

train_losses = []

def adjust_learning_rate(current_iteration, max_iteration, lr_min=0, lr_max=0.001, warmup_iteration=500):
    lr = 0.0
    if current_iteration <= warmup_iteration:
        lr = lr_max * current_iteration / warmup_iteration
    else:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((current_iteration - warmup_iteration) / max_iteration * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch,save_image = False):
    global iterations
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    losses = 0
    ssim_losses = 0
    mmd_losses = 0
    print(f'epoch : {epoch + 1}/{epochs}')
    for feature, scimg, label, ds in tqdm(traindataloader):
        iterations += 1
        adjust_learning_rate(current_iteration=iterations, max_iteration=max_iterations, lr_min=0, lr_max=base_lr)
        feature = feature.float()
        scimg = scimg.float()
        feature, scimg, ds = feature.to(device), scimg.to(device), ds.to(device)
        optimizer.zero_grad()
        feature_out, im_out, z, cls = model(feature)
        feature_out = feature_out.view(feature_out.size(0), feature_out.size(1), 14, 14)  # (b, c, t) ->(b, c, 14, 14)
        ssim_loss = 1 - criterion(im_out, scimg)  # ssim loss
        mmd_loss = 2 * mmd(z, ds,feature,feature_out) 
        train_loss = ssim_loss + mmd_loss
        train_loss.backward()
        optimizer.step()

        losses += train_loss.data.cpu()
        ssim_losses += ssim_loss.data.cpu()
        mmd_losses += mmd_loss.data.cpu()
    # Save the train images 
    if (epoch+1) % 5 == 0 and save_image:
        save_image(scimg, os.path.join('./img', f"{epoch}_real.jpg"), nrow=10, padding=2, pad_value=255)  # origin_images
        save_image(im_out, os.path.join('./img', f"{epoch}_model.jpg"), nrow=10, padding=2, pad_value=255) # model train images

    # Save the model weights 
    if (epoch+1)==300:
        torch.save(model.state_dict(), os.path.join(model_exit, model_name + str(epoch + 1).zfill(3) + ".mdl"))

    losses = losses / len(traindataloader)
    ssim_losses = ssim_losses / len(traindataloader)
    mmd_losses = mmd_losses / len(traindataloader)

    train_losses.append(losses.item())
    print('total_loss : {:.6f}, ssim_loss : {:.6}, mmd_loss : {:.6}'.format(losses, ssim_losses, mmd_losses))

if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
    # save loss curve image
    plt.figure()
    plt.plot(range(1, epoch + 2), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join('./img', 'train_loss_curve.png'))

