#!/usr/bin/env python3

import os
import sys
import json
import math
import time
import pprint as pp
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


from config import PARAMS_CONFIG
from models import EncoderSeq, QDecoder

from pack_model import build_model, get_tgt_entropy

from problems.pack2d.render import render


from trainer import train_epoch, full_eval, epoch_logger



from utils import (
    get_params,
    set_up_env,
    logger,
    log_graph,
    get_scheduler,
    get_grad_requiring_params,
    load_checkpoint,
    save_checkpoint)


def launch(env_params,
           model_params,
           problem_params,
           adapt_span_params,
           optim_params,
           trainer_params,
           rl_params):
    

    # print args and prepare directory and logger
    parameters_dict = locals()
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(model_params["seed"])
    # 预处理数据以及训练模型
    # ...
    # ...
    # print parameters
    for params_key, params_val in parameters_dict.items():
        print(params_key)
        pp.pprint(params_val)

    writer_name = "{}".format(env_params['run_name'])
    run_name = "{}_{}".format(env_params['run_name'], time.strftime("%Y%m%dT%H%M%S"))
    save_dir = os.path.join(
        env_params['output_dir'],
        "{}_{}_{}_{}".format(problem_params['problem_type'],"gaijin2", model_params['block_size'],model_params["seed"]),
        run_name
    )

    os.makedirs(save_dir)

    

    checkpoint_file = os.path.join(save_dir, 'checkpoint.pt')


    # Save arguments so exact configuration can always be found
    with open(os.path.join(save_dir, "args.json"), 'w') as fp:
    	json.dump(parameters_dict, fp)
        
    
    
    
    logger.configure(dir=save_dir, format_strs=os.getenv('OPENAI_LOG_FORMAT', 'log,csv').split(','))

    if not trainer_params['no_tensorboard']:
        tb_writer = SummaryWriter("runs/Dec30_12-21-39_china4-run"+str(model_params["seed"]))
    else:
        tb_writer = None

    # ENV and MODEL
    set_up_env(env_params)
    device = env_params['device']

    target_entropy = get_tgt_entropy(
        problem_params['problem_type'], 
        model_params['block_size'], 
        rl_params['tgt_entropy'],
        problem_params['p_options']
        ).to(device)


    modules = build_model(
        device, 
        problem_params, 
        model_params,
        # adapt_span_params
    )

    # show model size
    get_grad_requiring_params(modules)
    # print(modules)
    
    critic_params = [param for name, param in modules['critic'].named_parameters() if 'module.log_alpha' not in name]
    # OPTIMIZER AND SCHEDULER
    optimizer = optim.Adam([
                {'params': modules['actor'].parameters()},
                {'params': critic_params, 'lr': optim_params['critic_lr']},
                {'params': modules['critic'].module.log_alpha, 'lr': optim_params['critic_lr']}
            ], lr=optim_params['actor_lr'])


    lambda1 = lambda epoch: min(1, epoch / optim_params['lr_warmup'])

    # end_lr = 1e-1
    # start_lr = 1e-7
    # lr_find_epochs = trainer_params['nb_iter']/2
    # search_lambda = lambda epoch: math.exp(epoch * math.log(end_lr / start_lr) / (lr_find_epochs * model_params['block_size']))


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # warm up scheduler for both two groups             
    # scheduler = get_scheduler(optimizer, optim_params['lr_warmup'])

    iter_init = load_checkpoint(
        trainer_params['checkpoint_path'], modules, optimizer, scheduler)


    for epoch in tqdm(range(iter_init, trainer_params['nb_iter'])):
        # print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], run_name))
        # t_sta = time.time() # in seconds
        state, values, returns, losses, entropy, grad_norms, log_alpha = train_epoch(
            modules,
            optimizer,
            scheduler,
            problem_params,
            device,
            target_entropy,
            **model_params, **trainer_params, **optim_params, **rl_params)

        # for resume and render
        if epoch % trainer_params['checkpoint_interval'] == 0:
            # log_graph.save_train_graph(state, epoch, save_dir)
            save_checkpoint(checkpoint_file, epoch, modules, optimizer, scheduler)
        #     # with torch.no_grad():
            #     modules.eval()
            #     trainer_params['full_eval_mode'] = True
            #     t_sta = time.time() # in seconds

            #     state, values, returns, losses, entropy, _, _ = train_epoch(
            #             modules, 
            #             optimizer, 
            #             scheduler, 
            #             problem_params, 
            #             device, 
            #             target_entropy,
            #             **model_params, **trainer_params, **optim_params, **rl_params)

            #     gap_ratio = state.get_gap_ratio()
            #     avg_gap_ratio = gap_ratio.mean().item()

            #     elapsed = time.time() - t_sta

            #     print("Finished evaluation with gap ratio {}, took {} s".format(avg_gap_ratio, time.strftime('%H:%M:%S', time.gmtime(elapsed))))  

        # for monitor
        epoch_logger(epoch, state, values, returns, losses, entropy, grad_norms, log_alpha, optimizer,
                     tb_writer, trainer_params['log_interval'], run_name)

        
    # perform a evaluation after training
    
    with torch.no_grad():
        modules.eval()
        trainer_params['full_eval_mode'] = True
        t_sta = time.time() # in seconds

        state, values, returns, losses, entropy, _, _ = train_epoch(
                modules, 
                optimizer, 
                scheduler, 
                problem_params, 
                device, 
                target_entropy,
                **model_params, **trainer_params, **optim_params, **rl_params)

        gap_ratio = state.get_gap_ratio()
        avg_gap_ratio = gap_ratio.mean().item()

        elapsed = time.time() - t_sta

        print("Finished evaluation with gap ratio {}, took {} s".format(avg_gap_ratio, time.strftime('%H:%M:%S', time.gmtime(elapsed))))    





if __name__ == '__main__':
    launch(**get_params(params_config=PARAMS_CONFIG))




