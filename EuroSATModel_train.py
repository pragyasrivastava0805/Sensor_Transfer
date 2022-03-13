import numpy as np
import matplotlib.pyplot as plt
import torch
import torchgeo
from torchgeo.datasets import EuroSAT
import torch.nn as nn
import argparse
from PIL import Image
import time
import sys
import os

from torch import Tensor
from torch.nn import functional as F
import torch.optim as optim

import torchvision
import logging
from model import *
from torchvision import datasets, models, transforms
from utils import SaveBestModel
import glob
import utis
import tqdm
from torch.utils.data import DataLoader
from typing import Dict


class Normalize(torch.nn.Module):
    r"""Append normalized difference index as channel to image tensor.

    Computes the following index:

    .. math::

       \text{NDI} = \frac{A - B}{A + B}

    .. versionadded:: 0.2
    """

    
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        


  

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute and append normalized difference index to image.

        Args:
            sample: a sample or batch dict

        Returns:
            the transformed sample
        """
        sample["image"] = sample["image"].float()
        data_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        sample["image"] = data_transform(sample["image"])
        sample["label"] = sample["label"]
        return sample

parser = argparse.ArgumentParser(description="Training a Model on EuroSAT")
parser.add_argument('-b', '--batch-size', type=int, default=100, help='the size of the mini-batches when inferring features')
parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
parser.add_argument('-d','--data_path',type = str,default='../data/EuroSAT')
parser.add_argument('-l','--learning_rate',type=float,default=0.025)
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--save', type=str, default='EXP-ab1-', help='experiment name')
parser.add_argument('--epochs',type=int,default=600, help='Number of Epochs')
args = parser.parse_args()
data_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


train_dataset = EuroSAT(args.data_path,split='train',bands=("B04","B03","B02"),transforms=Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),download=True,checksum=False)
val_dataset = EuroSAT(args.data_path,split='val',bands=("B04","B03","B02"),transforms=Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),download=True,checksum=False)

data_loader_params = dict(batch_size = args.batch_size, shuffle = True, num_workers = 1, pin_memory = True)

trainloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
testloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, pin_memory = True)




save_best_model = SaveBestModel()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utis.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = resnet50().to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
optimizer = torch.optim.SGD(model.parameters(),args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/voyager-volume/EuroSAT_model/final_model.pth')

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('/voyager-volume/EuroSAT_model/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/voyager-volume/EuroSAT_model/loss.png')

def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    len_dataloader = len(trainloader)
    print(len_dataloader)
    counter = 0
    train_iter = iter(trainloader)
    for i, batch in enumerate(trainloader):
        counter += 1
        image = batch["image"]
        labels = batch["label"]
        logging.info("ok")
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    test_iter = iter(testloader)
    with torch.no_grad():
        for i, batch in enumerate(testloader):
    
            counter += 1
            
            image, labels = batch["image"], batch["label"]
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

for epoch in range(args.epochs):
    logging.info('epoch %d /epochs %e', epoch, args.epochs)
    train_epoch_loss, train_epoch_acc = train(model, trainloader, optimizer, criterion)

    valid_epoch_loss, valid_epoch_acc = validate(model, testloader,  criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

    save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion)
    print('-'*50)

save_model(args.epochs, model, optimizer, criterion)
save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('TRAINING COMPLETE')
        





