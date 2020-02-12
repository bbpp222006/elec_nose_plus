import os
import torch
import numpy as np
import random
import argparse
import random
import re
from  sklearn import preprocessing
import matplotlib.pyplot as plt
from itertools import groupby

from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data
from torch.autograd import Variable

from data_loader.data_loader import ElecNoseDataSet,ElecNoseDataLoader
from models.lstm import BLSTM
import utils


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_()#全连接层参数初始化





def train(crnn, train_loader, criterion):

    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    loss_avg = utils.averager()


    for i_batch,(x_train_batch, y_train_batch, x_train_batch_lengths) in enumerate(train_loader):
        x_train_batch_pack = rnn_utils.pack_padded_sequence(x_train_batch,
                                                          x_train_batch_lengths, batch_first=True)

        x_train_batch_pack = x_train_batch_pack.to(device)
        y_train_batch_encode, lengths = converter.encode(y_train_batch)
        # y_train_batch = converter.decode(y_train_batch_encode)
        preds = crnn(x_train_batch_pack)
        # out_pad, out_len = rnn_utils.pad_packed_sequence(preds, batch_first=True)
        out_pad = preds
        preds_size = torch.IntTensor([out_pad.size(0)] * len(y_train_batch))
        cost = criterion(out_pad, y_train_batch_encode, preds_size, lengths)

        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)
        print('cost',cost)

        if i_batch%10==0:

        # 绘图测试
            torch.save({'state_dict': crnn.state_dict()}, 'model/checkpoint.pth.tar')
            crnn.eval()
            train_preds = crnn(x_train_batch_pack,dev = True)
            utils.draw(train_preds[:,0,:],x_train_batch[0,:,:],y_train_batch_encode[0],ax1)
            for x_test_batch, y_test_batch, x_test_batch_lengths in val_loader:
                x_test_batch_pack = rnn_utils.pack_padded_sequence(x_test_batch,
                                                                    x_test_batch_lengths, batch_first=True)
                x_test_batch_pack = x_test_batch_pack.to(device)
                test_preds = crnn(x_test_batch_pack,dev = True)
                y_test_batch_encode, lengths = converter.encode(y_test_batch)
                utils.draw(test_preds[:, 0, :], x_test_batch[0, :, :], y_test_batch_encode[0], ax2)
                break
            plt.ylim(-0.5, 6)
            plt.pause(0.1)
            plt.show()
            ax2.cla()
            ax1.cla()
            crnn.train()
        # if (i_batch + 1) % 10 == 0:
        #     print('[%d/%d] Loss: %f' %
        #           ( i_batch, len(train_loader), loss_avg.val()))
        #     loss_avg.reset()

def main(crnn, train_loader, val_loader, criterion, optimizer):

    crnn = crnn.to(device)
    criterion = criterion.to(device)
    for e in range(10):
        train(crnn, train_loader, criterion)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 第一步：构造dataset

    elec_nose_dataset = ElecNoseDataSet()
    split = int(len(elec_nose_dataset)*0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(elec_nose_dataset, [split, len(elec_nose_dataset)-split])
    # 第二步：构造dataloader
    train_loader = ElecNoseDataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = ElecNoseDataLoader(val_dataset, batch_size=1, shuffle=True)


    # 构造损失函数
    criterion = torch.nn.CTCLoss(reduction='mean')
    #构造字符转换
    class_set = elec_nose_dataset.get_num_class()
    print(class_set)
    num_class = len(class_set)
    converter = utils.strLabelConverter(class_set)
    # 权重初始化




    blstm = BLSTM(num_class=num_class)
    if os.path.exists('model/checkpoint.pth.tar'):
        print('发现存在模型，加载中')
        checkpoint = torch.load('model/checkpoint.pth.tar')
        blstm.load_state_dict(checkpoint['state_dict'])
    else:
        blstm.apply(weights_init)


    optimizer = torch.optim.Adam(blstm.parameters(), lr=1e-5)
    # 绘图
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    main(blstm, train_loader, val_loader, criterion, optimizer)