import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from data_loader.data_loader import ElecNoseDataSet
import torch.nn.utils.rnn as rnn_utils


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    signal = [sq[0] for sq in data]
    signal = rnn_utils.pad_sequence(signal, batch_first=True, padding_value=0)
    return signal, [sq[1] for sq in data],data_length

# 第一步：构造dataset
dataset = ElecNoseDataSet()
# 第二步：构造dataloader
dataloader = DataLoader(dataset, batch_size=5, shuffle=True,collate_fn=collate_fn)

# 第三步：对dataloader进行迭代
for epoch in range(2):  # 只查看两个epoch
    for x_train_batch, y_train_batch, x_train_batch_lengths in dataloader:
        x_train_batch_pack = rnn_utils.pack_padded_sequence(x_train_batch,
                                                      x_train_batch_lengths, batch_first=True)
        # out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
        print(x_train_batch.size())
        print(x_train_batch_pack[0].size(),x_train_batch_pack[1])
        print(y_train_batch)
        print("-----------------------------------")