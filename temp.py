import os
import torch
import numpy as np
import random
import argparse
import random
import re

from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data
from torch.autograd import Variable

from data_loader.data_loader import ElecNoseDataSet
from models.lstm import BLSTM
import utils

import torch.nn as nn
from torch.nn import functional as F

y = torch.rand((10,4,6), requires_grad=True)
print(y)
# 指定在哪一个维度上进行Softmax操作，比如有两个维度：[batch, feature],
# 第一个维度为batch，第二个维度为feature,feature为一个三个值的向量[1, 2, 3],
# 则我们指定在第二个维度上进行softmax操作，则[1, 2, 3] => [p1, p2, p3]的概率值
# 因为y只有一个维度，所以下面指定的是在dim=0,第一个维度上进行的操作
p = F.softmax(y, dim=-1)

print(p)