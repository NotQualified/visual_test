import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        #input size: 28 * 28 -> 24 * 24 -> 12 * 12 -> 8 * 8 -> 4 * 4
        self.feat_ext = nn.Sequential(
                                    nn.Conv2d(1, 32, 5),
                                    nn.PReLU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 5),
                                    nn.PReLU(),
                                    nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
                                    nn.Linear(4 * 4 * 64, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 10))
    
    def forward(self, x):
        feat = self.feat_ext(x)
        feat = feat.view(-1, self.NumInstances(feat))
        return self.classifier(feat)

    def NumInstances(self, t):
        ret = 1
        for ele in t.size()[1: ]:
            ret *= ele
        return ret

