#!/usr/bin/env python
# coding: utf-8



#!/usr/bin/env python3

# command-line arguments with their default values

PARAMS_CONFIG = {
    # env-specific
    'env_params': {
        '--run-name': {
            'type': str,
            'default': 'run',
            'help': 'run name',
            'dest': 'run_name'
        },
        '--output-dir': {
            'type': str,
            'default': 'outputs',
            'help': 'output dir',
            'dest': 'output_dir'
        },
        
    },
    'rl_params': {
        '--ent-coef': {
                'type': float,
                'default': 3e-3,
                'help': 'entropy coefficient',
                'dest': 'ent_coef'
            },
        '--soft-temp': {
            'type': float,
            'default': 5e-3,
            'help': 'soft temperature of entropy regularization',
            'dest': 'soft_temp'
        },
        '--gamma': {
            'type': float,
            'default': 1,
            'help': 'reward discount factor',
            'dest': 'gamma'
        },
        '--nsteps': {
            'type': int,
            'default': 10,
            'help': 'GAE rolling out steps',
            'dest': 'nsteps'
        },
        '--lam': {
            'type': float,
            'default': 0.98,
            'help': 'lam for General Advantage Estimation',
            'dest': 'lam'
        },
        '--target-entropy': {
            'type': float,
            'default': -0.6,
            'help': 'position target entropy for entropy regularization',
            'dest': 'tgt_entropy'
        },
    },
    # model-specific
    'model_params': {
        '--hid-sz': {
            'type': int,
            'default': 128,
            'help': 'hidden size (i.e. model size)',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 512,
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--encoder-layers': {
            'type': int,
            'default': 3,
            'help': 'number of layers',
            'dest': 'encoder_layers'
        },
        '--decoder-layers': {
            'type': int,
            'default': 3,
            'help': 'number of layers',
            'dest': 'decoder_layers' 
        },
        '--critic-encoder-layers': {
            'type': int,
            'default':2,
            'help': 'number of layers',
            'dest': 'c_encoder_layers'
        },
        '--critic-decoder-layers': {
            'type': int,
            'default': 3,
            'help': 'number of layers',
            'dest': 'c_decoder_layers' 
        },
        '--block-sz': {
            'type': int,
            'default': 20,
            'help': 'block size '
                    '(the length of sequence to process in parallel)',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 8,
            'help': 'number of self-attention heads',
            'dest': 'nb_heads'
        },
        '--attn-span': {
            'type': int,
            'default': 20,
            'help': 'length of the attention span',
            'dest': 'attn_span'
        },
        '--dropout': {
            'type': float,
            'default': 0.4,
            'help': 'dropout rate of ReLU and attention',
            'dest': 'dropout'
        },
        '--normalization': {
            'type': str,
            'default': 'batch',
            'help': 'Normalization type,'
                    'batch or instance or layer',
            'dest': 'normalization'
        },
        '--head-hid': {
            'type': int,
            'default': 128,
            'help': 'head hidden dim(select and rotate)',
            'dest': 'head_hidden'
        },
        '--head-hid-pos': {
            'type': int,
            'default': 512,
            'help': 'position head hidden dim',
            'dest': 'head_hidden_pos'
        },
        '--seed': {
            'type': int,
            'default': 1,
            'help': 'number of boxes in each instance',
            'dest': 'seed'
        },
    },
    # problem
    'problem_params':{
        '--problem-type': {
            'type': str,
            'default': 'pack3d',
            'help': 'problem type (2d or 3d)',
            'dest': 'problem_type'
        },
        '--online': {
            'action': 'store_true',
            'default': False,
            'help': 'on-line packing',
            'dest': 'on_line'
        },
        '--noquery': {
            'type': bool,
            'default': False,
            'help': 'no query model',
            'dest': 'no_query'
        },
        '--block-num': {
            'type': int,
            'default': 1,
            'help': 'number of boxes in each instance',
            'dest': 'block_num'
        },
        '--position-options': {
            'type': int,
            'default': 128,
            'help': 'position options',
            'dest': 'p_options'
        },
        '--size-p1': {
            'type': float,
            # 'default': 0.02, # 2d
            'default': 2, # 3d
            'help': 'box mean for normal distribution sampling'
                    'box size high bound for uniform distribution sampling',
            'dest': 'size_p1'
        },
        '--size-p2': {
            'type': float,
            # 'default': 0.4, # 2d
            'default':8, # 3d
            'help': 'box variance for normal distribution sampling'
                    'box size high bound for uniform distribution sampling',
            'dest': 'size_p2'
        },
        '--data-distribution': {
            'type': str,
            'default': 'uniform',
            'help': 'Data distribution to use during training',
            'dest': 'distribution'
        },
    },
    # optimization-specific
    'optim_params': {
        '--actor-lr': {
            'type': float,
            'default': 4e-5,
            'help': 'actor learning rate',
            'dest': 'actor_lr'
        },
        '--critic-lr': {
            'type': float,
            'default': 1e-4,
            'help': 'critic learning rate',
            'dest': 'critic_lr'
        },
        '--lr-warmup': {
            'type': int,
            'default': 100,
            'help': 'linea+y increase LR from 0 '
                    'during first lr_warmup updates'
                     'warmup_epochs=lr_warmup/(block_size/nsteps)',
            'dest': 'lr_warmup'
        },
        '--grad-clip': {
            'type': float,
            'default': 5,
            'help': 'clip gradient of each module parameters by a given '
                    'value',
            'dest': 'grad_clip'
        },
    },
    # trainer-specific
    'trainer_params': {
        '--batch-sz': {
            'type': int,
            'default': 128,
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--niter': {
            'type': int,
            'default': 100000,
            'help': 'number of iterations to train',
            'dest': 'nb_iter'
        },
        '--log-interval': {
            'type': int,
            'default': 5,
            'help': 'number of epoch per command-line print log',
            'dest': 'log_interval'
        },
        '--checkpoint-interval': {
            'type': int,
            'default': 200,
            'help': 'number of epoch per checkpoint',
            'dest': 'checkpoint_interval'
        },
        '--no-tensorboard': {
            'action': 'store_true',
            'default': False,
            'help': 'disable tensorboard.',
            'dest': 'no_tensorboard'
        },
        '--checkpoint': {
            'type': str,
            'default': 'outputs/pack3d_gaijin2_20/run_20221229T060149/checkpoint.pt',
            'help': 'path to save/load model',
            'dest': 'checkpoint_path'
        },
        '--full-eval-mode': {
            'action': 'store_true',
            'default': False,
            'help': 'do evaluation on the whole validation and the test data',
            'dest': 'full_eval_mode'
        },
    },
    # adaptive attention span specific params
    'adapt_span_params': {
        '--adapt-span': {
            'action': 'store_true',
            'default': False,
            'help': 'enable adaptive attention span',
            'dest': 'adapt_span_enabled'
        },
        '--adapt-span-loss': {
            'type': float,
            'default': 0,
            'help': 'the loss coefficient for span lengths',
            'dest': 'adapt_span_loss'
        },
        '--adapt-span-ramp': {
            'type': int,
            'default': 16,
            'help': 'ramp length of the soft masking function',
            'dest': 'adapt_span_ramp'
        },
        '--adapt-span-init': {
            'type': float,
            'default': 0,
            'help': 'initial attention span ratio',
            'dest': 'adapt_span_init'
        },
        '--adapt-span-cache': {
            'action': 'store_true',
            'default': False,
            'help': 'adapt cache size as well to reduce memory usage',
            'dest': 'adapt_span_cache'
        },
    },
}

