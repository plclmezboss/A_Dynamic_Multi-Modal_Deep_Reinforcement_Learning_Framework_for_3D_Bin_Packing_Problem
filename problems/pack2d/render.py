#!/usr/bin/env python
# coding: utf-8


import time
import torch




BIN_HEIGHT = 150.0
DRAW_SCALE = 3.0

def get_render_data(state):
    
    # (batch, graph_size, n_feature)
    graphs = state.get_graph()
    heights = state.get_height()
    gap_sizes = state.get_gap_size()
    gap_ratios = state.get_gap_ratio()
    
    
    return graphs, heights, gap_sizes, gap_ratios

# graph (2*block_size, 4)

def render(graph, height, gap_size, gap_ratio, sleep=0):
    from problems.pack2d.viewer import Viewer

    screen_width = 2000
    screen_height = 1600
    
    viewer = Viewer(screen_width, screen_height)
    viewer.window.clear()

    min_y, _ = torch.min(graph[:,3], -1) # (1)

    print('min y: ', min_y.data)

    delta_height = height - min_y

    graph[:,3] -= min_y

    print('delta height: ', delta_height.data)

    scale = DRAW_SCALE * (BIN_HEIGHT / delta_height)
    
    viewer.set_scale(scale)
    viewer.draw_background(2*scale)
    
    
    # viewer.draw_text(height, gap_size, gap_ratio)

    for i, row in enumerate(graph):
        # print("i, row:", i, row)
        if row[0] == 0 or row[1] == 0:
            continue
        viewer.add_geom(row, i)
        if sleep>0:
            viewer.render() 
            time.sleep(sleep)

        
    viewer.draw_top_line(height)
    viewer.render() 
    
    return viewer

