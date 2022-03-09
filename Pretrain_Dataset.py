import torch
import random
import numpy as np
from   NILM_Dataset import *


class Pretrain_Dataset(NILMDataset):
    def __init__(self, x, y, status, window_size=480, stride=30, mask_prob=0.2):
        self.x = x
        self.y = y
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

        tokens  = []
        labels  = []
        on_offs = []

        #TODO: Optimize 
        for i in range(len(x)):
            prob = random.random()
            if prob < self.mask_prob:
                prob = random.random()
                if prob < 0.8:
                    tokens.append(-1)
                elif prob < 0.9:
                    tokens.append(np.random.normal())
                else:
                    tokens.append(x[i])

                labels.append(y[i])
                on_offs.append(status[i])
            else:
                tokens.append(x[i])
                temp = np.array([-1])
                labels.append(temp)
                on_offs.append(temp)
        return  torch.Tensor(tokens).unsqueeze(0), torch.Tensor(labels), torch.Tensor(on_offs)
