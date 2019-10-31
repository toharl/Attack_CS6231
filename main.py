'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import random
import numpy as np


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
'''this code is modified from the version of https://github.com/kuangliu/pytorch-cifar :

I manually change the lr during training:

0.1 for epoch [0,150)
0.01 for epoch [150,250)
0.001 for epoch [250,350)

Resume the training with python main.py --resume --lr=0.01
'''
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--PATH', default='Checkpoints/', type=str, help='path to save model')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--N', default=6, type=int, help='number of datasets')
parser.add_argument('--epochs', default=150, type=int, help='number of datasets')
parser.add_argument('--attack', default=False, type=bool, help='our attack of duplicates')
#parser.add_argument('--filter', default=True, type=bool, help='bool filter')
parser.add_argument('--poison', default=False, type=bool, help='poison attack')
parser.add_argument('--f', default=1.0, type=float, help='fraction to take from the samples when we use filter')
parser.add_argument('--d_a', default=0, type=int, help='d_a is the size of the attacker dataset')
args = parser.parse_args()

attack = args.attack
poison = args.poison
print()
print('========== Experiment params =========')
print('attack:', attack, ',poison:', poison, ',f:', args.f )
print()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_loss = 0 #best test loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# # Make attacker images
num_samples = int(50000 / (args.N - 1))#num samples in one dataset
if args.d_a > 0:
    num_samples = args.d_a
images, labels = next(iter(trainloader))

tensor_list = [images[0]]*num_samples
attacker_images = torch.stack(tensor_list)

dup_class = labels[0].item()
print('attacker duplicates the class ', dup_class)

attacker_labels = torch.zeros([num_samples],dtype=torch.int32)
attacker_labels.fill_(dup_class)
if poison:
    attacker_labels.fill_(3)
    print('poison attack')
    print(attacker_labels)


print('=========attacker dataset shape:===========')
print(attacker_images.shape)
print(attacker_labels.shape)
print()

import pdb
#pdb.set_trace()


class AttackerDataset(Dataset):
    """Attacker dataset."""

    def __init__(self, attacker_labels, attacker_images, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attacker_labels = attacker_labels
        self.attacker_images = attacker_images
        self.transform = transform

    def __len__(self):
        return self.attacker_labels.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if self.transform:
        #     sample = self.transform(sample)

        return (self.attacker_images[0], self.attacker_labels[0].item())

if attack or poison:
    attacker_dataset = AttackerDataset(attacker_labels=attacker_labels,
                                            attacker_images=attacker_images)
    # print('======attacker dataset')
    # print(len(attacker_dataset))
    # print(attacker_dataset[7])
    # for i in range(len(attacker_dataset)):
    #     print(i, attacker_dataset[i])


    concat = torch.utils.data.ConcatDataset([trainset, attacker_dataset])
    trainloader = DataLoader(concat, batch_size=128, shuffle=True, num_workers=2)


#print([trainloader, attacker_dataset])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


N = args.N
PATH = args.PATH
PATH = str(attack) + '_' + str(N) + '_f' + str(args.f) + '_pois' + str(args.poison) + '_da' + str(args.d_a)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(PATH+'_checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(PATH+'_checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


f = args.f

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



import pdb
#pdb.set_trace()



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        #pdb.set_trace()

        loss_each = loss#.detach() ? #https://discuss.pytorch.org/t/loss-for-each-sample-in-batch/36200
        x = int(f * len(loss_each))
        top_loss = loss_each.sort()[0][0:x] #sort and take f fraction of all -- small loss
        loss = torch.mean(top_loss)

        loss.backward() #loss : tensor(2.5326, device='cuda:0', grad_fn=<NllLossBackward>)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    global best_loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss = torch.mean(loss)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    calc_test_loss = test_loss/(batch_idx+1)
    if acc > best_acc:
        print('Saving to'+ str(PATH)+'_checkpoint ..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(PATH+'_checkpoint'):
            os.mkdir(PATH+'_checkpoint')
        torch.save(state, PATH+'_checkpoint/ckpt.pth')
        best_acc = acc
        best_loss = calc_test_loss


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)

print()
print('========== Experiment params =========')
print('our attack:', attack, ',poison:', poison, ',f:', args.f )
print('lr:', args.lr, 'd_a:', args.d_a)
print('acc:', best_acc)
print('loss:', best_loss)