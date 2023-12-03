#!/usr/bin/env python
# coding: utf-8

import time
from problems.pack3d.viewer import Viewer



def get_render_data(state):
    graphs = state.get_graph()
    gap_sizes = state.get_gap_size()
    gap_ratios = state.get_gap_ratio()
    heights = state.get_height()
    orders = state.get_order()

    return graphs, heights, gap_sizes, gap_ratios, orders


def render(graph, height, gap_size, gap_ratio, sleep=0):
    width = 2
    height = 2
    length = 2

    viewer = Viewer(width, height, length)

    graph = graph.numpy()
    # print(graph)
    # for i in range(20):
    #     row = graph[19-i]
    #     # print(row)
    #     viewer.add_geom(row)
    for row in graph:
        # print("row:", row)
        viewer.add_geom(row)


    # print(1)