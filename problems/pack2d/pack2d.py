#!/usr/bin/env python
# coding: utf-8


from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.pack2d.state_pack2d import StatePack2D

from utils import sample_truncated_normal, generate_normal


class Pack2DUpdate(Dataset):

    def __init__(self, block_size=20, batch_size=128, block_num=10, size_p1=0.4, size_p2=10.0, distribution='normal', online=False, **kargs):
        super(Pack2DUpdate, self).__init__()

        assert distribution is not None, "Data distribution must be specified for problem"

        if distribution == 'normal':
            self.data = [generate_normal(shape=(1, 2), mu=size_p1, sigma=size_p2, a=0.02, b=2.0)
                         for i in range(batch_size * block_size * block_num)]
        else:
            assert distribution == 'uniform'
            self.data = [torch.FloatTensor(1, 2).uniform_(
                size_p1, size_p2) for i in range(batch_size * block_size * block_num)]
            if not online:
                self.data = sorted(
                    self.data, key=lambda x: x[0][0].item() * x[0][1].item(), reverse=True)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class Pack2D(object):
    NAME = 'pack2d'

    @staticmethod
    def make_dataset(*args, **kwargs):
        return Pack2DUpdate(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePack2D(*args, **kwargs)
