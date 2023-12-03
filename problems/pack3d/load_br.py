#!/usr/bin/env python
# coding: utf-8

# In[269]:


import os
import numpy as np
from itertools import cycle


# In[260]:


# h=(w_0*L+l_0*W)/(w_0+l_0)
# w = w_0/W l=l_0/L 
def scale_inst_(inst, bin_size):
    inst[:,2] /= (inst[:,0]*bin_size[1] + inst[:,1]*bin_size[0])/(inst[:,0]+inst[:,1])
    inst[:,0] /= bin_size[0]
    inst[:,1] /= bin_size[1]
    return inst


# In[261]:


def read_instance_(data_cycle):
    boxes = []
    instance_num = np.asarray(list(map(int, next(data_cycle))))
    bin_size = np.asarray(list(map(int, next(data_cycle))))
    box_num = int(next(data_cycle)[0])
    
    for _ in range(box_num):
        boxes.append(np.asarray(list(map(int, next(data_cycle)))))
        
    boxes = np.asarray(boxes)
    # print(boxes)
    
    box_sizes = boxes[:,[1,3,5]]
    box_num = boxes[:,-1]
    inst = np.repeat(box_sizes, box_num, axis=0)
    
    # we drop bin size!
    return inst, bin_size


# In[270]:


def read_ds(file):
    values = []
    instances = []
    bin_sizes = []
    
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            values.append(line.split())
            
    data_cycle = cycle(values)
    
    for inst in range(int(next(data_cycle)[0])):
        inst, bin_size = read_instance_(data_cycle)
        inst = inst.astype(float)
        bin_size = bin_size.astype(float)
        instances.append(inst)
        bin_sizes.append(bin_size)
        
    instances = [scale_inst_(inst, bin_size) for inst, bin_size in zip(instances, bin_sizes)]
    instances = np.vstack(instances)
        
    return instances


# In[275]:


def get_br_ds(path, graph_size=200, batch_size=32):
    
    insts1 = read_ds(os.path.join(path, "br1.txt"))
    insts2 = read_ds(os.path.join(path, "br2.txt"))
    insts3 = read_ds(os.path.join(path, "br3.txt"))
    insts4 = read_ds(os.path.join(path, "br4.txt"))
    insts5 = read_ds(os.path.join(path, "br5.txt"))
    insts6 = read_ds(os.path.join(path, "br6.txt"))

    insts8 = read_ds(os.path.join(path, "br8.txt"))
    insts9 = read_ds(os.path.join(path, "br9.txt"))

    training_ds = np.vstack([insts1,insts2,insts3,insts4,insts5,insts6,insts8,insts9])
    test_ds = read_ds(os.path.join(path, "br7.txt"))
    
    divide_size = graph_size*batch_size
    
    b_n = training_ds.shape[0]//divide_size
    training_ds = training_ds[0:divide_size*b_n , :]
    
    b_n = test_ds.shape[0]//graph_size
    test_ds = test_ds[0:graph_size*b_n , :]
    
    return training_ds, test_ds






