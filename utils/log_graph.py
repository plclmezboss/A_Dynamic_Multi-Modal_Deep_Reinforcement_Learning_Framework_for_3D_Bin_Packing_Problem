#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
import torch
from problems.pack2d.render import get_render_data


def save_train_graph(state, epoch, save_dir):

    dataframe = _get_graph(state)
    dataframe.to_csv(os.path.join(save_dir, 'epoch-{}.csv'.format(epoch)))


def save_validate_graph(state, save_dir):
    dataframe = _get_graph(state)
    dataframe.to_csv(os.path.join(save_dir, 'validate.csv'))

def _get_graph(state):

    # (batch, graph, 4) (batch)
    render_data = get_render_data(state)

    # only save 10 examples
    def clip(data): return data[:10].data.cpu().numpy()
    # graph(n*5) others(n)
    clipped_data = list(map(clip, render_data))

    # min gap ratio
    min_gap_ratio, min_index = torch.topk(
        render_data[3], 10, -1, largest=False)

    min_graphs = torch.index_select(
        render_data[0], 0, min_index).data.cpu().numpy()
    min_gap_sizes = torch.index_select(
        render_data[2], 0, min_index).data.cpu().numpy()
    min_heights = torch.index_select(
        render_data[1], 0, min_index).data.cpu().numpy()
    min_gap_ratio = min_gap_ratio.data.cpu().numpy()

    min_render_data = [min_graphs, min_heights, min_gap_sizes, min_gap_ratio]

    df_min = list_to_df(min_render_data)
    df_random = list_to_df(clipped_data)

    df_save = pd.concat([df_min, df_random], axis=0)

    return df_save


def list_to_df(clipped_data):

    df = []

    graphs = clipped_data[0]

    for i, row in enumerate(graphs):
        #(20, 4) + ()
        graph_df = pd.DataFrame(row.transpose())
        graph_df['heights'] = clipped_data[1][i]
        graph_df['gap_sizes'] = clipped_data[2][i]
        graph_df['gap_ratios'] = clipped_data[3][i]
        df.append(graph_df)

    df = pd.concat(df)
    return df
