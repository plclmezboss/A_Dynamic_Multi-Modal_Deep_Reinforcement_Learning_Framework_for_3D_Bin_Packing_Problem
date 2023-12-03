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
from problems import Pack3D
from problems.pack3d.render import render, get_render_data


parser = argparse.ArgumentParser()

parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
parser.add_argument('--from_train', action='store_true', help='load from training example')
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
    if os.path.splitext(filename)[1] == '.csv' and os.path.splitext(filename)[0].split("-")[0] == 'epoch'
)
graph_filename = os.path.join(load_path, 'epoch-{}.csv'.format(epoch))
graph_filename = os.path.join(load_path, 'epoch-{}.csv'.format(opts.epoch))


print('  [*] Loading data from {}'.format(graph_filename))

data_frame = pd.read_csv(graph_filename)

indexs = np.arange(0, data_frame.shape[0], 6)

graph_size = data_frame.shape[1] - 4

# (batch*4, 20)
graphs = data_frame.iloc[:, 1:graph_size + 1].to_numpy()
assert graphs.shape[1] == graph_size

# data to cpu
# (batch, 4, 20)
plot_graphs = torch.from_numpy(graphs.reshape((graphs.shape[0] // 6, 6, graph_size)).swapaxes(1, 2))
heights = torch.from_numpy(data_frame.iloc[:, graph_size + 1].take(indexs).to_numpy())
gap_sizes = torch.from_numpy(data_frame.iloc[:, graph_size + 2].take(indexs).to_numpy())
gap_ratios = torch.from_numpy(data_frame.iloc[:, graph_size + 3].take(indexs).to_numpy())
# orders = torch.from_numpy(data_frame.iloc[:, graph_size+4].take(indexs).to_numpy())
# print(orders)
# orders = plot_graphs[:, :, 7]
# plot_graphs = plot_graphs[:, :, 0:7]

print("average height: ", heights.mean())
print("average gap_ratio: ", gap_ratios.mean())

print("rendering...")
# print(graphs[0])
# orders = orders.unsqueeze(-1).expand_as(plot_graphs).long()

# ranked_graph = plot_graphs.gather(1, orders)

draw_index = opts.index
print("height: ", heights[draw_index])
print('gap_sizes: ', gap_sizes[draw_index])
print('gap_ratios: ', gap_ratios[draw_index])

window = render(plot_graphs[draw_index], heights[draw_index], gap_sizes[draw_index], gap_ratios[draw_index], sleep=1)

# Plot the results


# Plot the results
while True:
    print("rendered")
    time.sleep(5)
