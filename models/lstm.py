import torch.nn as nn
import torch.nn.functional as F
import torch

class SequenceWise(nn.Module):
    '''调整输入满足module的需求，因为多次使用，所以模块化构建一个类
    适用于将LSTM的输出通过batchnorm或者Linear层
    '''
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        '''
        Args:
            x :    PackedSequence
        '''
        x, batch_size_len = x.data, x.batch_sizes
        #x.data:    sum(x_len) * num_features
        x = self.module(x)
        x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchSoftmax(nn.Module):
    '''
    The layer to add softmax for a sequence, which is the output of rnn
    Which state use its own softmax, and concat the result
    '''
    def forward(self, x):
        #x: seq_len * batch_size * num
        if not self.training:
            seq_len = x.size()[0]
            return torch.stack([F.softmax(x[i], dim=-1) for i in range(seq_len)], 0)
        else:
            return x

class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=True, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)


    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)

        # self.rnn.flatten_parameters()
        return x


class BLSTM(nn.Module):
    #                   种类      输入维度                               dropout
    def __init__(self, num_class, ni=6, rnn_hidden_size=256, leakyRelu=False, dropout=0.1):
        super(BLSTM, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        fc = nn.Sequential(nn.BatchNorm1d(2 * rnn_hidden_size),
                                nn.Linear(2 * rnn_hidden_size, num_class, bias=False,))
        self.fc = SequenceWise(fc)
        # self.log_softmax = BatchSoftmax()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.lstm = nn.Sequential(BatchRNN(ni, rnn_hidden_size,batch_norm = False),
                                  BatchRNN(rnn_hidden_size*2, rnn_hidden_size))

    def forward(self, input,dev=False):
        x = self.lstm(input)
        x = self.fc(x)  # 全连接，准备分类
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
        if dev:
            out = F.softmax(x , dim=-1)
            return out  # 如果是验证集，需要同时返回x计算loss和out进行wer的计算
        out = F.log_softmax(x, dim=-1)
        return out

#
# a = BLSTM(13)
# print(a)