#!/usr/bin/env python
# coding: utf-8


import torch
from typing import NamedTuple
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.functions import move_to

class PackAction():
    # (batch, 1)

    def __init__(self, batch_size, device):
        self.index = torch.zeros(batch_size, 1, device=device)
        self.x = torch.empty(
            batch_size, 1, device=device).fill_(-2)  # set to -2
        self.y = torch.empty(batch_size, 1, device=device).fill_(-2)
        self.z = torch.empty(batch_size, 1, device=device).fill_(-2)
        self.rotate = torch.zeros(batch_size, 1,device=device)
        self.updated_shape = torch.empty(batch_size, 3, device=device)
        self.sp = torch.FloatTensor(batch_size,  1).fill_(1)
        self.sp=move_to(self.sp,device)

        '''
        0: no rotate
        1: (x,y,z) -> (y,x,z)
        2: (x,y,z) -> (y,z,x)
        3: (x,y,z) -> (z,y,x)
        4: (x,y,z) -> (z,x,y)
        5: (x,y,z) -> (x,z,y)
        '''

    def set_index(self, selected):
        self.index = selected

    def set_rotate(self, rotate):
        self.rotate = rotate

    def set_shape(self, length, width, height):
        # (batch, 3)
        self.updated_shape = torch.stack([length, width, height], dim=-1)

    def get_shape(self):
        return self.updated_shape

    def set_pos(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_packed(self):
        return torch.cat((self.sp,self.updated_shape, self.x, self.y, self.z), dim=-1)

    def reset(self):
        self.__init__(self.index.size(0))

    def __call__(self):
        return {'index': self.index,
                'rotate': self.rotate,
                'x': self.x,
                'y': self.y}

    def __len__(self):
        return self.index.size(0)


def push_to_tensor_alternative(tensor, x):
    return torch.cat((tensor[:, 1:, :], x), dim=1)


class StatePack3D():

    def __init__(self, batch_size, block_size, device, position_size=128):

        # if online:
        #     self.boxes = torch.zeros(batch_size, 1, 3, device=device)
        # else:
        #     self.boxes = torch.zeros(batch_size, block_size, 3, device=device)

        # if cache_size is None:
        #     cache_size = block_size

        # self.online = online
        # self.instance_size = instance_size
        self.device = device
        # self.i = 0
        self.batch_size=batch_size
        # {length| width| height| x| y| z}
        self.packed_state = torch.zeros(
            batch_size, block_size, 7, dtype=torch.float, device=device)
        # self.packed_state_cache = torch.zeros(
        #     batch_size, cache_size, 6, dtype=torch.float, device=device)
        # self.packed_cat = torch.cat(
        #     (self.packed_state_cache, self.packed_state), dim=1)

        self.packed_rotate = torch.zeros(
            batch_size, block_size, 1, dtype=torch.int64, device=device)

        # self.boxes_volume = torch.zeros(
        #     batch_size, dtype=torch.float, device=device)
        self.total_rewards = torch.zeros(
            batch_size, dtype=torch.float, device=device)
        # self.skyline = torch.zeros(
        #     batch_size, position_size, position_size, dtype=torch.float, device=device)
        self.action = PackAction(batch_size, device=device)
        # self.z=[]
        self.container_size = [10, 10, 8*block_size]
        self.blocks_num = block_size
        self.block_dim = 3

        self.container = np.zeros([batch_size,10,10,8*block_size]).astype(int)
        self.positions = np.zeros((batch_size,self.blocks_num, self.block_dim)).astype(int)
        self.reward_type = "C-soft"
        self.stable = np.zeros((batch_size,self.blocks_num),dtype=bool)
        self.valid_size = [0]*batch_size
        self.empty_size = [0]*batch_size
        self.heightmap = np.zeros([batch_size,10,10]).astype('int')
        self.index=0
    def put_reward(self, reward):
        self.total_rewards += reward

    def get_rewards(self):
        return self.total_rewards

    # def get_mask(self):
    #
    #     mask_array = torch.from_numpy(
    #         np.tril(np.ones(20), k=-1).astype('bool_')).to(self.packed_state.device)
    #
    #     block_size = self.packed_state.size(1)
    #
    #     remain_steps = self.instance_size + block_size - self.i
    #     assert remain_steps > 0, 'over packed!!!'
    #     if remain_steps // block_size == 0:
    #         mask_num = block_size - remain_steps
    #     else:
    #         mask_num = 0
    #
    #     return mask_array[mask_num]

    def update_env(self, batch,batch_size,block_size,device):

        # if self.online:
        #     self.boxes = new_box

        # batch_size, block_size, box_state_size = self.boxes.size()
        #
        # all_index = torch.arange(
        #         block_size, device=self.boxes.device).repeat(batch_size, 1)
        #
        # mask = (
        #         all_index != self.action.index).unsqueeze(-1).repeat(1, 1, box_state_size)
        # # selected_box (batch, block-1, box_state_size)
        # remaining_boxes = torch.masked_select(
        #         self.boxes, mask).view(batch_size, -1, box_state_size)
        #
        # self.boxes = torch.cat((new_box, remaining_boxes), dim=1)
        # self.z.append(new_box[0])
        sp_initial=torch.FloatTensor(batch_size,block_size,1).fill_(0)
        sp_initial=move_to(sp_initial,device)
        position_initial=torch.FloatTensor(batch_size,block_size,3).fill_(0)
        position_initial=move_to(position_initial,device)
        self.packed_state=torch.cat([sp_initial,batch,position_initial],dim=2)
    def update_select(self, selected):
        self.action.set_index(selected)
        box_length, box_width, box_height = self._get_action_box_shape()

        self.action.set_shape(box_length, box_width, box_height)

    def _get_action_box_shape(self):

        # if self.online:
        #     box_length = self.boxes[:, :, 0].squeeze(-1).squeeze(-1)
        #     box_width = self.boxes[:, :, 1].squeeze(-1).squeeze(-1)
        #     box_height = self.boxes[:, :, 2].squeeze(-1).squeeze(-1)

        # else:

        select_index = self.action.index.long()

        box_raw_l = self.packed_state[:, :, 1].squeeze(-1)
        box_raw_w = self.packed_state[:, :, 2].squeeze(-1)
        box_raw_h = self.packed_state[:, :, 3].squeeze(-1)

        box_length = torch.gather(box_raw_l, -1, select_index).squeeze(-1)
        box_width = torch.gather(box_raw_w, -1, select_index).squeeze(-1)
        box_height = torch.gather(box_raw_h, -1, select_index).squeeze(-1)

        return box_length, box_width, box_height

    def update_rotate(self, rotate):

        self.action.set_rotate(rotate)

        # there are 5 rotations except the original one
        rotate_types = 5
        batch_size = rotate.size()[0]

        rotate_mask = torch.empty((rotate_types, batch_size), dtype=torch.bool)
        rotate_mask = move_to(rotate_mask, 'cuda')
        select_index = self.action.index.long()

        box_raw_x = self.packed_state[:, :, 1].squeeze(-1)
        box_raw_y = self.packed_state[:, :, 2].squeeze(-1)
        box_raw_z = self.packed_state[:, :, 3].squeeze(-1)

        # (batch)  get the original box shape
        box_length = torch.gather(box_raw_x, -1, select_index).squeeze(-1)
        box_width = torch.gather(box_raw_y, -1, select_index).squeeze(-1)
        box_height = torch.gather(box_raw_z, -1, select_index).squeeze(-1)

        for i in range(rotate_types):
            rotate_mask[i] = rotate.squeeze(-1).eq(i + 1)

        # rotate in 5 directions one by one
        # (x,y,z)->(y,x,z)
        # (x,y,z)->(y,z,x)
        # (x,y,z)->(z,y,x)
        # (x,y,z)->(z,x,y)
        # (x,y,z)->(x,z,y)
        inbox_length = box_length
        inbox_width = box_width
        inbox_height = box_height
        for i in range(rotate_types):

            if i == 0:
                box_l_rotate = box_width
                box_w_rotate = box_length
                box_h_rotate = box_height
            elif i == 1:
                box_l_rotate = box_width
                box_w_rotate = box_height
                box_h_rotate = box_length
            elif i == 2:
                box_l_rotate = box_height
                box_w_rotate = box_width
                box_h_rotate = box_length
            elif i == 3:
                box_l_rotate = box_height
                box_w_rotate = box_length
                box_h_rotate = box_width
            elif i == 4:
                box_l_rotate = box_length
                box_w_rotate = box_height
                box_h_rotate = box_width

            box_l_rotate = torch.masked_select(
                box_l_rotate, rotate_mask[i])
            box_w_rotate = torch.masked_select(
                box_w_rotate, rotate_mask[i])
            box_h_rotate = torch.masked_select(
                box_h_rotate, rotate_mask[i])

            inbox_length = inbox_length.masked_scatter(
                rotate_mask[i], box_l_rotate)
            inbox_width = inbox_width.masked_scatter(
                rotate_mask[i], box_w_rotate)
            inbox_height = inbox_height.masked_scatter(
                rotate_mask[i], box_h_rotate)
        self.packed_rotate[torch.arange(0, rotate.size(
            0)), select_index.squeeze(-1), 0] = rotate.squeeze(-1)

        self.action.set_shape(inbox_length, inbox_width, inbox_height)

    def update_pack(self):
        # batch_size = self.boxes.size(0)
        select_index = self.action.index.squeeze(-1).long().tolist()

        # z = self._get_z_skyline(x, y)
        x=torch.tensor(self.positions[:,self.index,0]).unsqueeze(-1).float()
        y=torch.tensor(self.positions[:,self.index,1]).unsqueeze(-1).float()
        z=torch.tensor(self.positions[:,self.index,2]).unsqueeze(-1).float()
        x=move_to(x,self.device)
        y=move_to(y,self.device)
        z=move_to(z,self.device)
        self.action.set_pos(x, y, z)

        packed_box = self.action.get_packed()
        # inbox_shape = self.action.get_shape()

        # self.boxes_volume += (inbox_shape[:, 0] *
        #                       inbox_shape[:, 1] * inbox_shape[:, 2]).squeeze(-1)

        # self.packed_state_cache = push_to_tensor_alternative(
        #     self.packed_state_cache, self.packed_state[:, 0:1, :])
        mid=self.packed_state.clone()
        self.packed_state=mid
        for i in range(self.batch_size):
            self.packed_state[i][select_index[i]]=packed_box[i]

        # self.packed_cat = torch.cat(
        #     (self.packed_state_cache, self.packed_state), dim=1)

        self.index += 1

    # def _get_z_skyline(self, x, y):
    #
    #     inbox_length = self.action.get_packed()[:, 0]
    #     inbox_width = self.action.get_packed()[:, 1]
    #     inbox_height = self.action.get_packed(
    #     )[:, 2].unsqueeze(-1).unsqueeze(-1)
    #
    #     position_size = self.skyline.size(1)
    #     batch_size = self.skyline.size(0)
    #
    #     in_back = torch.min(x.squeeze(-1), x.squeeze(-1) + inbox_length)
    #     in_front = torch.max(x.squeeze(-1), x.squeeze(-1) + inbox_length)
    #     in_left = torch.min(y.squeeze(-1), y.squeeze(-1) + inbox_width)
    #     in_right = torch.max(y.squeeze(-1), y.squeeze(-1) + inbox_width)
    #
    #     back_idx = ((in_back + 1.0) * (position_size / 2)
    #                 ).floor().long().unsqueeze(-1)
    #     front_idx = ((in_front + 1.0) * (position_size / 2)
    #                  ).floor().long().unsqueeze(-1)
    #     left_idx = ((in_left + 1.0) * (position_size / 2)
    #                 ).floor().long().unsqueeze(-1)
    #     right_idx = ((in_right + 1.0) * (position_size / 2)
    #                  ).floor().long().unsqueeze(-1)
    #
    #     mask_x = torch.arange(
    #         0, position_size, device=self.device).repeat(batch_size, 1)
    #     mask_y = torch.arange(
    #         0, position_size, device=self.device).repeat(batch_size, 1)
    #
    #     mask_back = mask_x.ge(back_idx)
    #     mask_front = mask_x.le(front_idx)
    #     mask_left = mask_y.ge(left_idx)
    #     mask_right = mask_y.le(right_idx)
    #
    #     mask_x = mask_back * mask_front
    #     mask_y = mask_left * mask_right
    #
    #     mask_x = mask_x.view(batch_size, position_size,
    #                          1).float()  # (batch, pos_size, 1)
    #     # (batch, 1, pos_size)
    #     mask_y = mask_y.view(batch_size, 1, position_size).float()
    #
    #     mask = torch.matmul(mask_x, mask_y)  # (batch, pos_size, pos_size)
    #
    #     masked_skyline = mask * self.skyline
    #     non_masked_skyline = (1 - mask) * self.skyline
    #
    #     max_z = torch.max(masked_skyline.view(
    #         batch_size, -1), 1)[0].unsqueeze(-1).float()
    #
    #     update_skyline = mask * \
    #         (max_z.unsqueeze(-1) + inbox_height) + non_masked_skyline
    #
    #     self.skyline = update_skyline
    #
    #     return max_z

    def _get_z(self, x, y):

        inbox_length = self.action.get_packed()[:, 0]
        inbox_width = self.action.get_packed()[:, 1]

        in_back = torch.min(x.squeeze(-1), x.squeeze(-1) + inbox_length)
        in_front = torch.max(x.squeeze(-1), x.squeeze(-1) + inbox_length)
        in_left = torch.min(y.squeeze(-1), y.squeeze(-1) + inbox_width)
        in_right = torch.max(y.squeeze(-1), y.squeeze(-1) + inbox_width)

        box_length = self.packed_cat[:, :, 0]
        box_width = self.packed_cat[:, :, 1]
        box_height = self.packed_cat[:, :, 2]

        box_x = self.packed_cat[:, :, 3]
        box_y = self.packed_cat[:, :, 4]
        box_z = self.packed_cat[:, :, 5]

        box_back = torch.min(box_x, box_x + box_length)
        box_front = torch.max(box_x, box_x + box_length)
        box_left = torch.min(box_y, box_y + box_width)
        box_right = torch.max(box_y, box_y + box_width)
        box_top = torch.max(box_z, box_z + box_height)

        in_back = in_back.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]])
        in_front = in_front.unsqueeze(-1).repeat(
            [1, self.packed_cat.size()[1]])
        in_left = in_left.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]])
        in_right = in_right.unsqueeze(-1).repeat(
            [1, self.packed_cat.size()[1]])

        is_back = torch.gt(box_front, in_back)
        is_front = torch.lt(box_back, in_front)
        is_left = torch.gt(box_right, in_left)
        is_right = torch.lt(box_left, in_right)

        is_overlaped = is_back * is_front * is_left * is_right
        non_overlaped = ~is_overlaped

        overlap_box_top = box_top.masked_fill(non_overlaped, 0)

        max_z, _ = torch.max(overlap_box_top, -1, keepdim=True)

        return max_z

    def get_boundx(self):
        batch_size = self.packed_state.size()[0]
        front_bound = torch.ones(
            batch_size, device=self.packed_state.device) - self.action.get_shape()[:, 0]

        return front_bound

    def get_boundy(self):

        batch_size = self.packed_state.size()[0]
        right_bound = torch.ones(
            batch_size, device=self.packed_state.device) - self.action.get_shape()[:, 1]

        return right_bound

    def get_height(self):


        return np.max(self.heightmap,axis=(1,2))

    def get_gap_size(self):

        bin_volumn = self.get_height() * 100.0

        gap_volumn = bin_volumn - self.valid_size
        gap_volumn=torch.tensor(gap_volumn)
        gap_volumn=move_to(gap_volumn,self.device)
        return gap_volumn

    def get_gap_ratio(self):

        bin_volumn = self.get_height() *100.0
        bin_volumn=torch.tensor(bin_volumn)
        bin_volumn=move_to(bin_volumn,self.device)
        ebselong=torch.tensor([0.0001])
        ebselong=move_to(ebselong,self.device)

        gap_ratio = self.get_gap_size() /( bin_volumn+ebselong)

        return gap_ratio

    def get_graph(self):
        return self.packed_cat
