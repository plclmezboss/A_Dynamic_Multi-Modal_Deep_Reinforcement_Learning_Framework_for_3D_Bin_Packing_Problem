#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import json


def get_row(path):
    arg_file = os.path.join(path, 'args.json')
    progress_file = os.path.join(path, 'progress.csv')

    with open(arg_file, 'r') as f:
        data = json.load(f)

    args_df = pd.DataFrame([data])

    prog_df = pd.read_csv(progress_file, index_col=0)

    last_20_line = prog_df.tail(5).mean(axis=0, skipna=True).reset_index()

    last_20_line.set_index('index', inplace=True)

    last_20_line = last_20_line.T

    test_row = args_df.join(last_20_line)

    return test_row


def get_test_folders(path):
    sub_folders = []
    dirs = []

    for r, d, f in os.walk(path):
        dirs[:] = [d for d in d if not d[0] == '.']
        for folder in dirs:
            # has epoch-x.pt
            folder_path = os.path.join(path, folder)
            if len(os.listdir(folder_path)) > 3:
                # print("go to ", folder_path)
                # for filename in os.listdir(folder_path):
                #    print("file", filename)
                epoch = max(
                    int(os.path.splitext(filename)[0].split("-")[1])
                    for filename in os.listdir(folder_path)
                    if os.path.splitext(filename)[1] == '.pt'
                )
                # print("epoch", epoch)

                if epoch > 2:
                    sub_folders.append(os.path.join(r, folder))
    return sub_folders


def select_coloum(df):
    drop_list = ['model', 'no_cuda', 'checkpoint_encoder',
                 'data_distribution', 'log_dir', 'output_dir',
                 'epoch_start', 'checkpoint_epochs', 'load_path',
                 'resume', 'no_tensorboard', 'no_progress_bar', 'use_cuda',
                 'save_dir', 'eval_only', 'normalization']

    df.drop(drop_list, axis=1, inplace=True)

    cols = list(df.columns.values)
    cols.pop(cols.index('run_name'))
    cols.pop(cols.index('gap_ratio'))
    cols.pop(cols.index('misc/step'))
    cols.pop(cols.index('problem'))
    cols.pop(cols.index('graph_size'))
    cols.pop(cols.index('epoch'))

    cols.pop(cols.index('lr_model'))
    cols.pop(cols.index('hidden_dim'))
    cols.pop(cols.index('batch_size'))

    df = df[['gap_ratio', 'misc/step', 'epoch', 'lr_model',
             'run_name', 'problem', 'graph_size',
             'hidden_dim', 'batch_size'] + cols]

    return df


def get_statistic(folders):
    test_list = []
    for folder in folders:
        test_list.append(get_row(folder))

    graph_stt = pd.concat(test_list, sort=False).reset_index()
    graph_stt = select_coloum(graph_stt)

    return graph_stt
