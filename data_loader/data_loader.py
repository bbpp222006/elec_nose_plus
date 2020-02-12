import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.preprocessing import StandardScaler


class ElecNoseDataSet(Dataset):
    def __init__(self):
        '''
        param：
            signal_path_list: 单个数据的文件位置
            signal_label_list: 单个数据的文件标，其实就是文件名称，str格式
        数据默认在根目录下的dataset中
        '''
        self.signal_path_list = []
        self.signal_label_list = []
        self.scaler = StandardScaler()
        for root, dirs, files in os.walk('dataset'):
            if files:
                for fn_txt in files:
                    abs_fn_txt_path = os.path.join(root, fn_txt)
                    fn, ext = os.path.splitext(fn_txt)
                    if ext == '.txt'and fn != '基线':
                        self.signal_label_list.append(fn)
                        self.signal_path_list.append(abs_fn_txt_path)
        assert len(self.signal_label_list) == len(self.signal_path_list)
        random_choice = random.sample(self.signal_path_list, int(0.1*len(self.signal_path_list)))
        random_signal = np.genfromtxt(random_choice[0], delimiter='	')[:, 1:]
        for i in random_choice[1:]:
            random_signal = np.vstack((random_signal, np.genfromtxt(i, delimiter='	')[:, 1:]))
        self.scaler.fit_transform(random_signal)

    def get_num_class(self):
        return set(self.signal_label_list+['基线'])


    def __len__(self):
        return len(self.signal_label_list)

    def __getitem__(self, idx):
        assert len(self.signal_label_list) == len(self.signal_path_list)
        signal = np.genfromtxt(self.signal_path_list[idx], delimiter='	')
        signal = torch.from_numpy(self.scaler.transform(signal[:, 1:]))
        signal_label = self.signal_label_list[idx]


        return signal.float(), signal_label

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    signal = [sq[0] for sq in data]
    signal = torch.nn.utils.rnn.pad_sequence(signal, batch_first=True, padding_value=0)
    return signal, [sq[1] for sq in data],data_length

class ElecNoseDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ElecNoseDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn


