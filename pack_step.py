#!/usr/bin/env python3


import torch
from torch import nn
import copy
from torch.nn import DataParallel
import torch.nn.functional as F
from utils.functions import move_to
import torch
import itertools
from matplotlib.path import Path
import numpy as np
from scipy.spatial import ConvexHull
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def calc_one_position_lb_greedy_3d(block, block_index, container_size, reward_type,
                                   container, positions, stable, heightmap, valid_size, empty_size):

    block_dim = len(block)
    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    # x coordinate
    hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
    hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x) - 1, axis=0)
    hm_diff_x = heightmap - hm_diff_x
    # y coordinate
    hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
    hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T) - 1, axis=1)
    hm_diff_y = heightmap - hm_diff_y

    # get the xy coordinates of all left-deep-bottom corners
    ems_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
    ems_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()
    ems_xy_list = []
    ems_xy_list.append([0, 0])
    for xy in ems_x_list:
        x, y = xy
        if y != 0 and [x, y - 1] in ems_x_list:
            if heightmap[x, y] == heightmap[x, y - 1] and \
                    hm_diff_x[x, y] == hm_diff_x[x, y - 1]:
                continue
        ems_xy_list.append(xy)
    for xy in ems_y_list:
        x, y = xy
        if x != 0 and [x - 1, y] in ems_y_list:
            if heightmap[x, y] == heightmap[x - 1, y] and \
                    hm_diff_x[x, y] == hm_diff_x[x - 1, y]:
                continue
        if xy not in ems_xy_list:
            ems_xy_list.append(xy)

    # sort by y coordinate, then x
    def y_first(pos):
        return pos[1]

    ems_xy_list.sort(key=y_first, reverse=False)

    # get ems_list
    for xy in ems_xy_list:
        x, y = xy
        if x + block_x > container_size[0] or \
                y + block_y > container_size[1]: continue
        z = np.max(heightmap[x:x + block_x, y:y + block_y])
        ems_list.append([x, y, z])

    # firt consider the most bottom, sort by z coordinate, then y last x
    def z_first(pos):
        return pos[2]

    ems_list.sort(key=z_first, reverse=False)

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # varients to store results of searching ems corners
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems = [False] * ems_num
    is_stable_ems = [False] * ems_num
    compactness_ems = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems = [0.0] * ems_num
    empty_ems = [empty_size] * ems_num
    under_space_mask = [[]] * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _y, _z):
        # check if the pos visited
        if [_x, _y, _z] in visited: return
        if _z > 0 and (container[_x:_x + block_x, _y:_y + block_y, _z - 1] == 0).all(): return
        visited.append([_x, _y, _z])
        if (container[_x:_x + block_x, _y:_y + block_y, _z] == 0).all():
            if not is_stable(block, np.array([_x, _y, _z]), container):
                if reward_type.endswith('hard'):
                    return
            else:
                is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _y, _z])
            heightmap_ems[index][_x:_x + block_x, _y:_y + block_y] = _z + block_z
            is_settle_ems[index] = True

    # calculate socres
    def calc_C_P_S(index):
        _x, _y, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        bbox_size = height * container_size[0] * container_size[1]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x + block_x, _y:_y + block_y, 0:_z]
        under_space_mask[index] = under_space == 0
        empty_ems[index] += np.sum(under_space_mask[index])
        if 'P' in reward_type:
            pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        # stability
        if 'S' in reward_type:
            stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
            stability_ems[index] = stable_num / (block_index + 1)

    # search positions in each ems
    X = int(container_size[0] - block_x + 1)
    Y = int(container_size[1] - block_y + 1)
    for ems_index, ems in enumerate(ems_list):
        # using buttom-left strategy in each ems
        heightmap_ems[ems_index] = heightmap.copy()
        X0, Y0, _z = ems
        for _x, _y in itertools.product(range(X0, X), range(Y0, Y)):
            if is_settle_ems[ems_index]: break
            check_position(ems_index, _x, _y, _z)
        if is_settle_ems[ems_index]: calc_C_P_S(ems_index)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # get the best ems
    ratio_ems = [c + p + s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the dynamic parameters
    _x, _y, _z = pos_ems[best_ems_index]
    container[_x:_x + block_x, _y:_y + block_y, _z:_z + block_z] = block_index + 1
    container[_x:_x + block_x, _y:_y + block_y, 0:_z][under_space_mask[best_ems_index]] = -1
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    heightmap = heightmap_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]

    return container, positions, stable, heightmap, valid_size, empty_size
def is_stable(block, position, container):
    '''
    check for 3D packing
    ----
    '''
    if (position[2]==0):
        return True
    x_1 = position[0]
    x_2 = x_1 + block[0] - 1
    y_1 = position[1]
    y_2 = y_1 + block[1] - 1
    z = position[2] - 1
    obj_center = ( (x_1+x_2)/2, (y_1+y_2)/2 )

    # valid points right under this object
    points = []
    for x in range(x_1, x_2+1):
        for y in range(y_1, y_2+1):
            if (container[x][y][z] > 0):
                points.append([x, y])
    if(len(points) > block[0]*block[1]/2):
        return True
    if(len(points)==0 or len(points)==1):
        return False
    elif(len(points)==2):
        # whether the center lies on the line of the two points
        a = obj_center[0] - points[0][0]
        b = obj_center[1] - points[0][1]
        c = obj_center[0] - points[1][0]
        d = obj_center[1] - points[1][1]
        # same ratio and opposite signs
        if (b==0 or d==0):
            if (b!=d): return False
            else: return (a<0)!=(c<0)
        return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )
    else:
        # calculate the convex hull of the points
        points = np.array(points)
        try:
            convex_hull = ConvexHull(points)
        except:
            # error means co-lines
            min_point = points[np.argmin( points[:,0] )]
            max_point = points[np.argmax( points[:,0] )]
            points = np.array( (min_point, max_point) )
            a = obj_center[0] - points[0][0]
            b = obj_center[1] - points[0][1]
            c = obj_center[0] - points[1][0]
            d = obj_center[1] - points[1][1]
            if (b==0 or d==0):
                if (b!=d): return False
                else: return (a<0)!=(c<0)
            return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )

        hull_path = Path(points[convex_hull.vertices])
        return hull_path.contains_point((obj_center))

def pack_step(modules, state,  problem_params):
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

    hm=torch.tensor(hm).float()
    hm=move_to(hm,state.device)
    actor_modules = modules['actor']

    actor_encoder_out = actor_modules['encoder'](state.packed_state)
    if state.index == 0:
        actor_encoder_out_select = torch.zeros(state.batch_size, 1, 128)
        actor_encoder_out_select = move_to(actor_encoder_out_select, state.device)
    else:
        actor_encoder_out_select = torch.masked_select(actor_encoder_out, state.packed_state[:, :, 0].unsqueeze(-1).bool())
        actor_encoder_out_select = actor_encoder_out_select.view(state.batch_size, state.index, 128)

    actor_encoderheightmap_out=actor_modules["encoderheightmap"](hm)
    q_select=torch.cat((actor_encoder_out_select,actor_encoderheightmap_out),dim=1)
    q_select=torch.mean(q_select,dim=1).unsqueeze(1)
    # if not state.online:
    #     # (batch, block, 1)
    s_out = actor_modules['s_decoder'](q_select, actor_encoder_out)
    select_mask=state.packed_state[:,:,0].bool()

    # select_mask = state.get_mask()
#         print(state.boxes, state.packed_state)
    s_log_p, selected = _select_step(s_out.squeeze(1), select_mask)

    # else:
    #     selected = torch.zeros(state.packed_state.size(0), device=state.packed_state.device)
    #     s_log_p = 0

    # select (batch)
    state.update_select(selected)
    # (batch, 2)
    selected=selected.expand(state.batch_size,128).unsqueeze(1)
    actor_encoder_out_rotation=torch.gather(actor_encoder_out,1,selected)

    q_rotation=torch.cat((actor_encoder_out_rotation,actor_encoderheightmap_out),dim=1)
    q_rotation=torch.mean(q_rotation,dim=1).unsqueeze(1)


    r_out = actor_modules['r_decoder'](q_rotation, actor_encoder_out).squeeze(1)

    r_log_p, rotation = _rotate_step(r_out.squeeze(-1))

    # rotation
    state.update_rotate(rotation)
    blocks=state.action.get_shape()


    for  i,j in enumerate(blocks):
        block=j.int().tolist()
        block_index=state.index
        container_size = state.container_size
        blocks_num =state.blocks_num
        block_dim = state.block_dim

        container = state.container[i]
        positions = state.positions [i]
        reward_type =state.reward_type
        stable = state.stable[i]
        valid_size = state.valid_size[i]
        empty_size =state.empty_size[i]
        heightmap = state.heightmap[i]
        state.container[i], state.positions [i],state.stable[i], state.heightmap[i], state.valid_size[i],state.empty_size[i]= calc_one_position_lb_greedy_3d(block,
                                                                                                         block_index,
                                                                                                         container_size,
                                                                                                         reward_type,
                                                                                                         container,
                                                                                                         positions,
                                                                                                         stable,
                                                                                                         heightmap,
                                                                                                         valid_size,
                                                                                                         empty_size)

    # if problem_params['problem_type'] == 'pack2d':
    #     p_position = state.action.get_shape().unsqueeze(1)
    #
    #     if not problem_params['no_query']:
    #
    #         p_out = actor_modules['p_decoder'](p_position, actor_encoder_out).squeeze(1)
    #
    #     else:
    #
    #         p_out = actor_modules['p_decoder'](q_rotation, actor_encoder_out).squeeze(1)
    #
    #     x_log_p, box_xs = _drop_step(p_out.squeeze(-1), state.get_boundx())
    #
    #     value, h_caches[1] = modules['critic'](state.boxes, state.packed_state, h_caches[1])
    #     value = value.squeeze(-1)
    #     # update location and finish one step packing
    #     state.update_pack(box_xs)
    #
    #     return s_log_p, r_log_p, x_log_p, value, h_caches
    # else:
    #
    #     p_position = state.action.get_shape().unsqueeze(1)
    #     q_position = state.action.get_shape().unsqueeze(1)
    #
    #
    #     if not problem_params['no_query']:
    #
    #         p_out = actor_modules['p_decoder'](p_position, actor_encoder_out).squeeze(1)
    #         q_out = actor_modules['q_decoder'](q_position, actor_encoder_out).squeeze(1)
    #     else:
    #
    #         p_out = actor_modules['p_decoder'](q_rotation, actor_encoder_out).squeeze(1)
    #         q_out = actor_modules['q_decoder'](q_rotation, actor_encoder_out).squeeze(1)

        # x_log_p, box_xs = _drop_step(p_out.squeeze(-1), state.get_boundx())
        # y_log_p, box_ys = _drop_step(q_out.squeeze(-1), state.get_boundy())

    value = modules['critic'](actor_encoderheightmap_out, actor_encoder_out)
    value = value.squeeze(-1).squeeze(-1)

    state.update_pack()

    return s_log_p, r_log_p, value




def _select_step(s_logits, mask):

    s_logits = s_logits.masked_fill(mask, -np.inf)

    s_log_p = F.log_softmax(s_logits, dim=-1)

    # (batch)
    selected = _select(s_log_p.exp()).unsqueeze(-1)

    # do not reinforce masked and avoid entropy become nan
    s_log_p = s_log_p.masked_fill(mask, 0)

    return s_log_p, selected


def _rotate_step(r_logits):

    r_log_p = F.log_softmax(r_logits, dim=-1)

    # rotate (batch, 1)
    rotate = _select(r_log_p.exp()).unsqueeze(-1)
    
    return r_log_p, rotate

def _select(probs, decode_type="sampling"):
    assert (probs == probs).all(), "Probs should not contain any nans"
    
    if decode_type == "greedy":
        _, selected = probs.max(-1)
    elif decode_type == "sampling":
        selected = probs.multinomial(1).squeeze(1)
    
    else:
        assert False, "Unknown decode type"
        
    return selected


