import torch
import torch.nn as nn
import torch.optim as optim
import basic.utils.utils as ut
from torch.utils.data import DataLoader
from models.model import BaseNet
from torchvision.datasets import MNIST
from torchvision import transforms

mean, std = 0.1307, 0.3081
batch_size = 64
epochs = 10

gpu_ids = [1, 2, 3]
ut.CudaDevices(gpu_ids)

train_dataset = MNIST('../data/MNIST', train = True, download = True, 
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((mean, ), (std, ))]))
test_dataset = MNIST('../data/MNIST', train = False, download = True, 
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((mean, ), (std, ))]))

train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(test_dataset, batch_size, False)

if_load = False
net = BaseNet()
try:
    checkpoint = {}
    basenet_state_dic = torch.load('./model_save/mnist_sd.pth')
    basenet_state_dic = ut.ModifyStateDict(basenet_state_dic)
    net.load_state_dict(basenet_state_dic)
    if_load = True
    net = net.cuda()
except BaseException:
    net = nn.DataParallel(net).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters())

if not if_load:
    for epoch in range(0, epochs):
        cur_loss, cur = 0, 0
        right, total = 0, 0
        for i, (data, label) in enumerate(train_loader):
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad() 
            output, _ = net(data)
            right += (torch.argmax(output, dim = 1) == label).sum().item()
            total += label.size()[0]
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("epoch:%d iteration:%d precision:%.5f" % (epoch, i, right / total))
                right, total = 0, 0

right, total = 0, 0
for i, (data, label) in enumerate(test_loader):
    data, label = data.cuda(), label.cuda()
    output, _ = net(data)
    right += (torch.argmax(output, dim = 1) == label).sum().item()
    total += label.size()[0]
    precision = right / total
print("final precision: ", precision)
torch.save(net.state_dict(), "./model_save/mnist_sd.pth")
