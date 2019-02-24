import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from models.model import BaseNet
from torchvision.datasets import MNIST
from torchvision import transforms


mean, std = 0.1307, 0.3081
batch_size = 64
epochs = 10

test_dataset = MNIST('../data/MNIST', train = False, download = True, 
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((mean, ), (std, ))]))

test_loader = DataLoader(test_dataset, batch_size, False)

net = torch.load('./model_save/mnist_adam_model.ckpt')

feat = []
labels = []

for i, (data, label) in enumerate(test_loader):
    data, label = data.cuda(), label.cuda()
    _, features = net(data, True)
    feat += features.cpu().detach().numpy().tolist()
    labels += label.cpu().detach().numpy().tolist()
    if len(feat) > 200:
        break
feat = np.array(feat)
labels = np.array(labels)
print(feat.shape)
print(labels.shape)

X_tsne = TSNE(n_components = 2, random_state = 33).fit_transform(feat)
ckpt_dir = "images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

plt.figure(figsize = (5, 5))
plt.subplot(111)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = labels, label = 'mnist')
plt.legend()

plt.savefig('images/mnist_adam_tsne.png', dpi = 120)
