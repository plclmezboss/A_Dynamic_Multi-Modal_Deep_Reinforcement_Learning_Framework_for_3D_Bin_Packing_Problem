#!/usr/bin/env python3

import os
import math
import argparse

import torch


def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--block-sz'
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)


def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config['dest']:
                namespace.__getattribute__(param_config['dest'])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }


##############################################################################
# ENVIRONMENT
##############################################################################

def _torch_distributed_init_process_group(local_rank):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print('my rank={} local_rank={}'.format(rank, local_rank))
    torch.cuda.set_device(local_rank)
    return {
        'rank': rank,
        'world_size': world_size,
    }

def set_up_env(env_params):
    if torch.cuda.is_available():
        env_params['device'] = torch.device('cuda')
    else:
        env_params['device'] = torch.device('cpu')


##############################################################################
# OPTIMIZER AND SCHEDULER
##############################################################################

def get_grad_requiring_params(model):
    nb_parameters = 0
    grad_requiring_params = []
    for param in model.parameters():
        if param.requires_grad:
            nb_parameters += param.numel()
            grad_requiring_params.append(param)
    print('nb_parameters={:.2f}M'.format(nb_parameters / 1e6))
    return grad_requiring_params





def get_scheduler(optimizer, lr_warmup):
    if lr_warmup > 0:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup))
    return None




##############################################################################
# CHECKPOINT
##############################################################################

def _load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    print('loading from a checkpoint at {}'.format(checkpoint_path))
    
    checkpoint_state = torch.load(checkpoint_path)
    iter_init = checkpoint_state['iter_no'] + 1  # next iteration
    model.load_state_dict(checkpoint_state['model'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])
    if 'scheduler_iter' in checkpoint_state:
        # we only need the step count
        scheduler.step(checkpoint_state['scheduler_iter'])
    return iter_init


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path and os.path.exists(checkpoint_path):
        return _load_checkpoint(checkpoint_path=checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler)
    return 0


def save_checkpoint(checkpoint_path, iter_no, modules,
                    optimizer, scheduler):
    if checkpoint_path:
        checkpoint_state = {
            'iter_no': iter_no,  # last completed iteration
            'model': modules.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint_state['scheduler_iter'] = scheduler.last_epoch
        torch.save(checkpoint_state, checkpoint_path)


