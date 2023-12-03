import math
import time
import random

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


import numpy as np

from utils import (
    move_to,
    load_problem, 
    logger, 
    explained_variance)

from problems.pack2d.render import render
from pack_step import pack_step


def train_epoch(
    modules, 
    optimizer, 
    scheduler, 
    problem_params, 
    device,
    target_entropy,
    batch_size, 
    block_size, 
    hidden_size, 
    **kargs):


    problem = load_problem(problem_params['problem_type'])

    dataupdate = problem.make_dataset(block_size=block_size, batch_size=batch_size, online=problem_params['on_line'], **problem_params)

    update_dataloader = DataLoader(dataupdate, batch_size=batch_size)

    # only encoder requires cache 
    # h_cache_a = [
    #     torch.zeros(
    #         batch_size,
    #         layer.attn.attn.get_cache_size(),
    #         hidden_size).to(device)
    #     for layer in modules['actor']['encoder'].module.layers]
    #
    # h_cache_c = [
    #     torch.zeros(
    #         batch_size,
    #         layer.attn.attn.get_cache_size(),
    #         hidden_size).to(device)
    #     for layer in modules['critic'].module.att_encoder.layers]

    # h_caches = [h_cache_a, h_cache_c]

    updatedata_iterator = iter(update_dataloader)
    # instance_size = len(updatedata_iterator)
    batch=next(updatedata_iterator)
    batch=move_to(batch,device)
    state = problem.make_state(
        batch_size,block_size, device)
    state.update_env(batch,batch_size,block_size,device)

    state, values, returns, losses, entropy, grad_norms = train_instance(
        modules, 
        optimizer, 
        scheduler, 
        # h_caches,
        block_size,
        state, 
        # updatedata_iterator,
        target_entropy,
        problem_params, 
        **kargs)

    # For TensorboardX logs
    alpha_tlogs = modules['critic'].module.log_alpha.clone()

    return state, values, returns, losses, entropy, grad_norms, alpha_tlogs 





def train_instance(
    modules, 
    optimizer, 
    scheduler, 
    # h_caches,
    block_size,
    state, 
    # updatedata_iterator,
    target_entropy,
    problem_params,
    **kargs):

    device = state.packed_state.device
    total_losses = torch.tensor([0,0,0,0], dtype=torch.float, device=device)
    total_entropy = torch.zeros(1, dtype=torch.float, device=device)

    # if box number is not inter times of nsteps, then we drop last several data
    # update_mb_number = int(len(updatedata_iterator) // kargs['nsteps'])
    update_mb_number=int(block_size//kargs["nsteps"])
    # pack the last block
    # if not state.online:
    for mb_id in tqdm(range(update_mb_number), disable=True):
        state, entropy, values, returns, losses, grad_norms = train_minibatch(
                modules, 
                optimizer, 
                scheduler, 
                # h_caches,
                state, 
                # updatedata_iterator,
                target_entropy, 
                device, 
                problem_params, 
                **kargs)

        total_entropy += entropy
        total_losses += losses

        # Here we want to pack the last block
    # last_mb_number = block_size // kargs['nsteps']

        # for mb_id in tqdm(range(last_mb_number), disable=True):
        #     state, entropy, values, returns, losses, grad_norms = train_minibatch(
        #         modules,
        #         optimizer,
        #         scheduler,
        #         h_caches,
        #         state,
        #         None,
        #         target_entropy,
        #         device,
        #         problem_params,
        #         **kargs)
        #
        # total_entropy += entropy
        # total_losses += losses
        # update_mb_number += last_mb_number

    # else:
    #     for mb_id in tqdm(range(update_mb_number), disable=True):
    #
    #         state, entropy, values, returns, losses, grad_norms = train_minibatch(
    #             modules,
    #             optimizer,
    #             scheduler,
    #             h_caches,
    #             state,
    #             updatedata_iterator,
    #             target_entropy,
    #             device,
    #             problem_params,
    #             **kargs)
    #
    #         total_entropy += entropy
    #         total_losses += losses


    average_entropy = total_entropy / update_mb_number
    average_losses = total_losses / update_mb_number

    return state, values, returns, average_losses, average_entropy, grad_norms



def train_minibatch(modules, 
    optimizer, 
    scheduler, 
    # h_caches,
    state, 
    # data_iterator,
    target_entropy, 
    device, 
    problem_params, 
    ent_coef, 
    full_eval_mode,
    **kargs):

    returns, advs, values, log_likelihoods, entropys = get_mb_data(
            modules,  state, device, problem_params, **kargs)

    alpha_loss = -1 * torch.mv((-1*entropys + target_entropy).detach(), modules['critic'].module.log_alpha).mean()

    # entropy_loss = -1 * ent_coef * entropys.mean()

    value_loss = F.mse_loss(values, returns.float().detach())

    advs = returns - values
    # Normalize the advantages
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    # do not backward critic in actor
    advantages = advs.detach()

    # Calculate loss (gradient ascent)  loss=A*log
    actor_loss = -1 * (advantages * log_likelihoods).mean()
    
    loss = actor_loss + value_loss + alpha_loss

    if not full_eval_mode:
        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, kargs['grad_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
    else:
        grad_norms = None

    losses = torch.tensor([actor_loss, value_loss, alpha_loss, loss], device=device)

    return state, entropys.mean(), values, returns, losses, grad_norms



def get_mb_data(modules, state, device, problem_params, gamma, nsteps, lam, soft_temp, **kargs):
    
    def sf01(arr):
        """
        swap and then flatten axes 0 and 1
        """
        s = arr.size()
        return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:]) 

    mb_rewards, mb_values, mb_log_likelihoods, mb_entropy = [],[],[],[]

    for _ in range(nsteps):
        # pack last block
        # if updatedata_iterator is None:
        #     if problem_params['problem_type'] == 'pack2d':
        #         batch = torch.zeros(state.packed_state.size(0), 1, 2, device=device)
        #     else:
        #         batch = torch.zeros(state.packed_state.size(0), 1, 3, device=device)
        #
        # else:
        #     try:
        #         batch = next(updatedata_iterator)
        #         batch = move_to(batch, device)
        #
        #         # print(batch[0])
        #     except StopIteration:
        #         print("-------------------------------------------------------------")
        #         print("No more data in the instance!!!")
        #         print("-------------------------------------------------------------")
        # # print('batch 0: ',batch[0])

        
        ll, entropy, value, reward = _run_batch(modules, state,  problem_params, soft_temp)
        mb_values.append(value)
        mb_log_likelihoods.append(ll)
        mb_rewards.append(reward)
        mb_entropy.append(entropy)

    # batch of steps to batch of roll-outs (nstep, batch)
    mb_rewards = torch.stack(mb_rewards)
    mb_values = torch.stack(mb_values)
    mb_log_likelihoods = torch.stack(mb_log_likelihoods)
    # (nstep, batch, 3)
    mb_entropys = torch.stack(mb_entropy)
    if (state.packed_state[:,:,0]==1).all():
        last_values=torch.zeros(state.batch_size,dtype=torch.float)
        last_values=move_to(last_values,state.device)
    else:

        hm = np.zeros((state.batch_size, 2, state.heightmap[0].shape[0], state.heightmap[0].shape[1])).astype(int)
        for i in range(state.batch_size):
            hm_diff_x = np.insert(state.heightmap[i], 0, state.heightmap[i][0, :], axis=0)
            hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x) - 1, axis=0)
            hm_diff_x = state.heightmap[i] - hm_diff_x
            # hm_diff_x = np.delete(hm_diff_x, 0, axis=0)
            # y coordinate
            hm_diff_y = np.insert(state.heightmap[i], 0, state.heightmap[i][:, 0], axis=1)
            hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T) - 1, axis=1)
            hm_diff_y = state.heightmap[i] - hm_diff_y
            # hm_diff_y = np.delete(hm_diff_y, 0, axis=1)
            # combine

            hm[i][0] = hm_diff_x
            hm[i][1] = hm_diff_y

        hm = torch.tensor(hm).float()
        hm = move_to(hm, state.device)
        actor_modules = modules['actor']

        actor_encoder_out = actor_modules['encoder'](state.packed_state)
        actor_encoderheightmap_out = actor_modules["encoderheightmap"](hm)
        last_values = modules['critic'](actor_encoderheightmap_out, actor_encoder_out)
        last_values = last_values.squeeze(-1).squeeze(-1)

    mb_returns = torch.zeros_like(mb_rewards)
    mb_advs = torch.zeros_like(mb_rewards)

    lastgaelam = 0

    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            nextvalues = last_values
        else:
            nextvalues = mb_values[t+1]

        delta = mb_rewards[t] + gamma * nextvalues - mb_values[t]
        mb_advs[t] = lastgaelam = delta + gamma * lam * lastgaelam

    # use return to supervise critic
    mb_returns = mb_advs + mb_values

    # (batch * nstep, )
    returns, advs, values, log_likelihoods, entropys = map(sf01, \
                (mb_returns, mb_advs, mb_values, mb_log_likelihoods, mb_entropys))


    return  returns, advs, values, log_likelihoods, entropys


def _run_batch(modules, state,  problem_params, soft_temp):

    # # update pack candidates for next packing step
    # state.update_env(batch)

    last_gap = state.get_gap_size()

    if problem_params['problem_type'] == 'pack2d':

        s_log_p, r_log_p, x_log_p, value, h_caches = pack_step(modules, state, h_caches, problem_params)
    
        actions = state.action()
        # position to discrete
        actions['x'] = ((actions['x'] + 1.0) * (x_log_p.size(1)/2)).round().long()

        # ll (batch, 1), entropy (batch)
        ll = _calc_log_likelihood(actions, s_log_p, r_log_p, x_log_p, state.online)
        entropys = _calc_entropy(s_log_p, r_log_p, x_log_p, state.online)

    elif problem_params['problem_type'] == 'pack3d':

        s_log_p, r_log_p, value= pack_step(modules, state, problem_params)

        actions = state.action()
        
        # actions['x'] = ((actions['x'] + 1.0) * (x_log_p.size(1)/2)).round().long()
        # actions['y'] = ((actions['y'] + 1.0) * (y_log_p.size(1)/2)).round().long()

        ll = _calc_log_likelihood_3d(actions, s_log_p, r_log_p)
        entropys = _calc_entropy_3d(s_log_p, r_log_p)

    # # ll (batch, 1), entropy (batch, 3)
    # ll = _calc_log_likelihood(actions, s_log_p, r_log_p, x_log_p, state.online)
    # entropys = _calc_entropy(s_log_p, r_log_p, x_log_p, state.online)

    # (batch)
    new_gap = state.get_gap_size()
    reward = (last_gap - new_gap)

    alpha = torch.exp(modules['critic'].module.log_alpha)


    # print('entropy: ', entropys.size(), "alpha: ", alpha.size())
    reward +=  torch.mv(entropys, alpha).detach()

    state.put_reward(reward)

    return ll, entropys, value, reward


def _calc_entropy(s_log, r_log, x_log, online):
    # log (batch, action_num)
    
    # S=-/sum_i (p_i \ln p_i)
    if online:
        s_entropy = torch.zeros(r_log.size(0), device=r_log.device)
    else:
        s_entropy = -1 * (s_log.exp() *s_log).sum(dim=-1)

    r_entropy = -1 * (r_log.exp() *r_log).sum(dim=-1)
    x_entropy = -1 * (x_log.exp() * x_log).sum(dim=-1)
    # print(r_entropy, r_entropy.size())
    # (batch, 3)
    entropys = torch.stack([s_entropy, r_entropy, x_entropy], dim=-1)
    # entropy = x_entropy

    assert not torch.isnan(entropys).any()

    return entropys

def _calc_entropy_3d(s_log, r_log):
    # log (batch, action_num)
    
    # S=-/sum_i (p_i \ln p_i)
    # if online:
    #     s_entropy = torch.zeros(r_log.size(0), device=r_log.device)

    s_entropy = -1 * (s_log.exp() *s_log).sum(dim=-1)

    r_entropy = -1 * (r_log.exp() * r_log).sum(dim=-1)
    # x_entropy = -1 * (x_log.exp() * x_log).sum(dim=-1)
    # y_entropy = -1 * (y_log.exp() * y_log).sum(dim=-1)

    # (batch)
    # entropy = s_entropy + r_entropy + x_entropy + y_entropy
    entropys = torch.stack([s_entropy, r_entropy], dim=-1)

    # entropy = x_entropy

    assert not torch.isnan(entropys).any()

    return entropys

def _calc_log_likelihood(actions, s_log, r_log, x_log, online):

    # actions (batch, 4)
    # log (batch, action_num)
    
    # (batch, 1)
    
    action_r = actions['rotate']
    action_x = actions['x']

    #(batch)
    if online:
        s_log_p = 0
    else:
        action_s = actions['index']
        s_log_p = s_log.gather(1, action_s).squeeze(-1)
        assert (s_log_p > -1000).data.all(), "log probability should not -inf, check sampling"

    r_log_p = r_log.gather(1, action_r).squeeze(-1)
    x_log_p = x_log.gather(1, action_x).squeeze(-1)

    
    assert (r_log_p > -1000).data.all(), "log probability should not -inf, check sampling"
    assert (x_log_p > -1000).data.all(), "log probability should not -inf, check sampling"
    
    log_likelihood = s_log_p+ r_log_p + x_log_p
    
    # print(s_log_p.mean(), r_log_p.mean(), x_log_p.mean())

    return log_likelihood

def _calc_log_likelihood_3d(actions, s_log, r_log):

    # actions (batch, 4)
    # log (batch, action_num)
    
    # (batch, 1)
    
    action_r = actions['rotate']
    action_x = actions['x']
    action_y = actions['y']
    #(batch)
    # if online:
    #     s_log_p = 0

    action_s = actions['index']
    s_log_p = s_log.gather(1, action_s).squeeze(-1)
    assert (s_log_p > -1000).data.all(), "log probability should not -inf, check sampling"

    r_log_p = r_log.gather(1, action_r).squeeze(-1)
    # x_log_p = x_log.gather(1, action_x).squeeze(-1)
    # y_log_p = y_log.gather(1, action_y).squeeze(-1)

    
    assert (r_log_p > -1000).data.all(), "log probability should not -inf, check sampling"
    # assert (x_log_p > -1000).data.all(), "log probability should not -inf, check sampling"
    # assert (y_log_p > -1000).data.all(), "log probability should not -inf, check sampling"

    
    log_likelihood = s_log_p+ r_log_p
    
    # print(s_log_p.mean(), r_log_p.mean(), x_log_p.mean())

    return log_likelihood


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped



# do full evaluation
def full_eval(
    modules, 
    optimizer, 
    scheduler, 
    h_caches, 
    problem, 
    init_dataloader, 
    update_dataloader, 
    device,
    ent_coef, 
    **kargs):
    
    modules.eval()

    return train_epoch(modules, optimizer, scheduler, problem_params, device,
            **model_params, **trainer_params, **optim_params, **rl_params)



def epoch_logger(epoch, state, values, returns, losses, entropy, grad_norms, log_alpha, optimizer, tb_writer, log_interval, run_name):

    total_gap = state.get_gap_size()
    gap_ratio = state.get_gap_ratio()
    rewards = state.get_rewards()
    
    avg_gap = total_gap.mean().item()
    avg_gap_ratio = gap_ratio.mean().item()
    var_gap_ratio = gap_ratio.var().item()
    avg_rewards = rewards.mean().item()
    min_gap = torch.min(gap_ratio)
    max_gap = torch.max(gap_ratio)

    grad_norms, grad_norms_clipped = grad_norms

    ev = explained_variance(values.detach().cpu().numpy(), returns.detach().cpu().numpy())

    # Log values to screen
    if epoch % log_interval == 0:
        print("state.packed_state:{}".format(state.packed_state))
        print('\nepoch: {}, run {}, avg_rewards: {}, gap_ratio: {}, var_gap_ratio: {}, ev: {}, loss: {}'.\
              format(epoch, run_name, avg_rewards, avg_gap_ratio, var_gap_ratio, ev, losses[3]))
        print('min gap ratio: {}, max gap ratio: {}'.format(min_gap, max_gap))
        print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
        print('grad_norm_c: {}, clipped_c: {}'.format(grad_norms[1], grad_norms_clipped[1]))
        
        print("entropy:{}".format(entropy))
        # print("heights:{}".format(state.get_height()))
        # print("z:{}".format(state.z))
    logger.logkv("epoch", epoch)
    logger.logkv("explained_variance", float(ev))
    logger.logkv('entropy', entropy.item())
        
    logger.logkv('actor_loss', losses[0].item())
    logger.logkv('value_loss', losses[1].item())
    logger.logkv('alpha_loss', losses[2].item())
    logger.logkv('avg_rewards', avg_rewards)
    logger.logkv('gap_ratio', avg_gap_ratio)
    logger.logkv('var_gap_ratio', var_gap_ratio)

    logger.dumpkvs()

    # Log values to tensorboard
    if tb_writer is not None:
        tb_writer.add_scalar('avg_rewards', avg_rewards, epoch)
        tb_writer.add_scalar('entropy', entropy, epoch)
        tb_writer.add_scalar('s_log_alpha', log_alpha[0].item(), epoch)
        tb_writer.add_scalar('r_log_alpha', log_alpha[1].item(), epoch)
        # tb_writer.add_scalar('p_log_alpha', log_alpha[2].item(), epoch)
        tb_writer.add_scalar('gap_ratio', avg_gap_ratio, epoch)
        tb_writer.add_scalar('var_gap_ratio', var_gap_ratio, epoch)

        tb_writer.add_scalar('min_gap', min_gap, epoch)
        tb_writer.add_scalar('explained_variance', float(ev), epoch)


        
        tb_writer.add_scalar('actor_loss', losses[0].item(), epoch)
        tb_writer.add_scalar('value_loss', losses[1].item(), epoch)
        tb_writer.add_scalar('alpha_loss', losses[2].item(), epoch)
        
        tb_writer.add_scalar('grad_norm', grad_norms[0], epoch)
        tb_writer.add_scalar('grad_norm_c', grad_norms[1], epoch)

        tb_writer.add_scalar('learnrate_pg0', optimizer.param_groups[0]['lr'], epoch)
        tb_writer.add_scalar('learnrate_pg1', optimizer.param_groups[1]['lr'], epoch)
