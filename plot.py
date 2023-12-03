#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import numpy as np
import torch
import time
import glob
import pandas as pd
import json


from torch.utils.data import DataLoader
from utils import load_model
from problems import Pack2D
from problems.pack2d.render import render, get_render_data




parser = argparse.ArgumentParser()

parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
# parser.add_argument('--from_val', action='store_true', help='load from training example')
parser.add_argument('--index', type=int, default=0, help="less than 10 is min, 11~20 is normal")
parser.add_argument('--epoch', type=int, help='choose the epoch to plot')

opts = parser.parse_args()

if opts.load_path is not None:
    load_path = opts.load_path
else:
    load_path = max(glob.iglob('outputs/pack2d_20/*'), key=os.path.getctime)


load_path = '/Users/phoenix/rcw/rl/packing/unversal/unversal_packing'


epoch = max(
            int(os.path.splitext(filename)[0].split("-")[1])
            for filename in os.listdir(load_path)
            if os.path.splitext(filename)[1] == '.csv' and os.path.splitext(filename)[0].split("-")[0]=='epoch'
        )
graph_filename = os.path.join(load_path, 'epoch-{}.csv'.format(epoch))
graph_filename = os.path.join(load_path, 'epoch-{}.csv'.format(opts.epoch))

    

print('  [*] Loading data from {}'.format(graph_filename))
print("draw the ", opts.index)

data_frame = pd.read_csv(graph_filename)

# index with step size 4
indexs = np.arange(0, data_frame.shape[0], 4)

# last 3 is statistic
graph_size = data_frame.shape[1]-4

#(batch*4, 20) the first column is csv index
graphs = data_frame.iloc[:,1:graph_size+1].to_numpy()
assert graphs.shape[1] == graph_size


# data to cpu
#(batch, 4, 20)
plot_graphs = torch.from_numpy(graphs.reshape((graphs.shape[0]//4, 4, graph_size)).swapaxes(1,2))
heights = torch.from_numpy(data_frame.iloc[:,graph_size+1].take(indexs).to_numpy())
gap_sizes = torch.from_numpy(data_frame.iloc[:,graph_size+2].take(indexs).to_numpy())
gap_ratios = torch.from_numpy(data_frame.iloc[:,graph_size+3].take(indexs).to_numpy())

print("average height: ", heights.mean().data)
print("average gap_ratio: ", gap_ratios.mean().data)

print("rendering...")
# print(graphs[0])


draw_index = opts.index

print("index height: ", draw_index, heights[draw_index])
print("index gap_ratio: ", draw_index, gap_ratios[draw_index])

window = render(plot_graphs[draw_index], heights[draw_index], gap_sizes[draw_index], gap_ratios[draw_index], sleep=0.2)

#for draw_index in range(20):
#    window = render(ranked_graph[draw_index], heights[draw_index], gap_sizes[draw_index], gap_ratios[draw_index], sleep=0.2)

# Plot the results


# Plot the results
while True:
    print("rendered")
    time.sleep(5)





