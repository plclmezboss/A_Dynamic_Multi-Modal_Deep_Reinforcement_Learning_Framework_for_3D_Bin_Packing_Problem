#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import Dataset
from problems.pack3d.load_br import get_br_ds
import torch
import os
from random import randint
from problems.pack3d.state_pack3d import StatePack3D

from utils import generate_normal

BIN_LENGTH = 50
BIN_WIDTH = 50
HALF_BIN_LENGTH = (BIN_LENGTH / 2)
HALF_BIN_WIDTH = (BIN_WIDTH / 2)


class Pack3DInit(Dataset):

    def __init__(self, block_size=20, batch_size=128, size_p1=2, size_p2=8, distribution='uniform', **kargs):
        super(Pack3DInit, self).__init__()

        assert distribution is not None, "Data distribution must be specified for problem"

        if distribution == 'normal':
            self.data = [generate_normal(shape=(block_size, 3), mu=size_p1, sigma=size_p2, a=0.02, b=2.0) for i in range(batch_size)]
        elif distribution == 'br':
            training_ds, _ = get_br_ds('.')
            self.data = torch.from_numpy(training_ds)
        else:
            assert distribution == 'uniform'
            self.data = [torch.randint(2,8,(block_size,3)).float() for i in range(batch_size)]
        
        self.size = len(self.data)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]

class Pack3DUpdate(Dataset):

    def __init__(self, block_size=20, batch_size=128, block_num=10, size_p1=0.4, size_p2=10.0, distribution='normal', **kargs):
        super(Pack3DUpdate, self).__init__()

        assert distribution is not None, "Data distribution must be specified for problem"

        if distribution == 'br':
            training_ds, _ = get_br_ds('./problems/pack3d/br/')
            # n(1*3) n=batch_size*block_size*block_num
            training_ds = torch.split(torch.from_numpy(training_ds).float(), 1, 0)

            num_samples = batch_size*block_size*block_num

            assert num_samples < len(training_ds), "dataset size is too small for current setting!!!"
            
            test_num = len(training_ds)//num_samples
            test_id = randint(0, test_num-1)

            self.data = training_ds[test_id*num_samples: test_id*num_samples + num_samples]

        elif distribution == 'normal':
            self.data = [generate_normal(shape=(1, 3), mu=size_p1, sigma=size_p2, a=0.02, b=2.0) for i in range(batch_size*block_size*block_num)]
        else:
            assert distribution == 'uniform'
            self.data = [torch.FloatTensor(block_size, 3).uniform_(size_p1, size_p2) for i in range(batch_size)]

        self.size = len(self.data)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

# one block zero data for last block feeding
class ZeroDataeset(Dataset):
    def __init__(self, block_size=20, batch_size=128, **kargs):
        super(ZeroDataeset, self).__init__()

        self.data = [torch.zeros(1, 3) for i in range(batch_size*block_size)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class Pack3D(object):
    NAME = 'pack3d'

    @staticmethod
    def make_dataset(*args, **kwargs):
        return Pack3DInit(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePack3D(*args, **kwargs)

# class Pack3dDataset(Dataset):
#     """docstring for Pack3dDataset"""

#     def __init__(self, size=20, num_samples=10000, offset=0, distribution=None):
#         super(Pack3dDataset, self).__init__()


#         # 8 max 1
#         # 20 max 0.4
#         # min_box_size = 0.1 * 10 / size
#         # max_box_size = 1.4 * 10 / size

#         min_box_size = 0.6
#         max_box_size = 1.2

#         self.data = [torch.FloatTensor(size, 3).uniform_(min_box_size, max_box_size) for i in range(num_samples)]

#         self.size = len(self.data)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         return self.data[idx]



