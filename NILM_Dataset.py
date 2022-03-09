import torch
import torch.utils.data as data_utils
import numpy as np


class NILMDataset(data_utils.Dataset):
    def __init__(self, x, y, status, window_size=480, stride=30):
        self.x           = x
        self.y           = y
        self.status      = status
        self.window_size = window_size
        self.stride      = stride

    def __len__(self):
        return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)

    def __getitem__(self,index):
        start_index = index * self.stride
        end_index   = np.min((len(self.x), index * self.stride + self.window_size))
        
        x        = self.padding_seqs(self.x[start_index: end_index])
        y        = self.padding_seqs(self.y[start_index: end_index])
        status   = self.padding_seqs(self.status[start_index: end_index])
        

        #####MAYBE DONT CONVERT TO TENSORS YET
        channels = self.y.shape[1]
        x        = torch.Tensor(x).view((1,-1))
        y        = torch.Tensor(y).view((channels,-1))
        status   = torch.Tensor(status).view((channels,-1))
        
        return x, y, status 