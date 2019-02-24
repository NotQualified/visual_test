import torch
import basic.utils.utils as ut
from models.model import BaseNet as bn
from tensorboardX import SummaryWriter

ut.CudaDevices([0, 1, 2, 3])

dummy_input = torch.randn(64, 1, 28, 28).cuda()

net = bn().cuda()

with SummaryWriter(comment = 'BaseNet', log_dir = 'logs') as w:
    w.add_graph(net, (dummy_input, ))
