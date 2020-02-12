import os
import torch
import numpy as np
import random
import argparse
import random
import re
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data
from torch.autograd import Variable

from data_loader.data_loader import ElecNoseDataSet, ElecNoseDataLoader
from models.lstm import BLSTM
from utils import *
import matplotlib.pyplot as plt







#
# a = torch.tensor([[0.2, 0.3, 0.5],
#                   [0.7, 0.3, 0.5],
#                   [0.7, 0.9, 0.5]
#                   ])
# b = torch.tensor([[0.2, 0.3, 0.5],
#                   [0.7, 0.3, 0.5],
#                   [0.7, 0.9, 0.5]
#                   ])
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# draw(a,b,2,ax)
