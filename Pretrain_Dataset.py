import torch
import random
import numpy as np
from   NILM_Dataset import *

class Pretrain_Dataset(NILMDataset):
    def __init__(self, x, y, status, window_size=480, stride=30, mask_prob=0.2):
        self.x           = x
        self.y           = y
        self.status      = status
        self.window_size = window_size
        self.stride      = stride
        self.mask_prob   = mask_prob

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index   = np.min((len(self.x), index * self.stride + self.window_size))
       
        x       = self.padding_seqs(self.x[start_index: end_index])
        y       = self.padding_seqs(self.y[start_index: end_index])
        status  = self.padding_seqs(self.status[start_index: end_index])

        for i in range(len(x)):
            prob = random.random()
            if prob<self.mask_prob:
                prob = random.random()
                x[i] = -1 if prob<0.8 else np.random.normal() if prob<0.9 else x[i]
            else:
                y[i]      = -1
                status[i] = -1

        x, y, status = [torch.Tensor(v).unsqueeze(0) for v in [x,y,status]]
        print('X.shape:')
        print(x.shape)
        print('Y.shape:')
        print(y.shape)
        return  x, y, status
