#!/usr/bin/env python


import torch
import copy 
import numpy as np
from typing import NamedTuple
from torch.utils.data import DataLoader
from tqdm import tqdm



class PackAction():
    # (batch, 1)
    
    def __init__(self, batch_size, device):
        self.index = torch.zeros(batch_size, 1, device=device)
        self.x = torch.empty(batch_size, 1, device=device).fill_(-2) # set to -2
        self.y = torch.empty(batch_size, 1, device=device).fill_(-2)
        self.rotate = torch.zeros(batch_size, device=device)
        self.updated_shape = torch.empty(batch_size, 2, device=device)

    def set_index(self, selected):
        self.index = selected
        
    def set_rotate(self, rotate):
        self.rotate = rotate

    def set_shape(self, width, height):
        # (batch, 2)
        self.updated_shape = torch.stack([width, height], dim=-1)

    def get_shape(self):
        return self.updated_shape


    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def get_packed(self):
        return torch.cat((self.updated_shape, self.x, self.y), dim=-1)

    def reset(self):
        self.__init__(self.index.size(0))
        
    def __call__(self):
        # (batch, 1)
        return {'index': self.index, 
                'rotate': self.rotate,
                'x': self.x}

    def __len__(self):
        return self.index.size(0)



def push_to_tensor_alternative(tensor, x):
    return torch.cat((tensor[:,1:,:], x), dim=1)



class StatePack2D():
    def __init__(self, batch_size, instance_size, block_size, device, position_size=128 ,online=False, cache_size=None):

        if online:
            self.boxes = torch.zeros(batch_size, 1, 2, device=device)
        else:
            self.boxes = torch.zeros(batch_size, block_size, 2, device=device)


        if cache_size is None:
            cache_size = block_size

        self.instance_size = instance_size
        self.device = device
        self.online = online
        self.i = 0 

        # {width| height| x| y}
        self.packed_state = torch.zeros(batch_size, block_size, 4, dtype=torch.float, device=device)
        self.packed_state_cache = torch.zeros(batch_size, cache_size, 4, dtype=torch.float, device=device)
        self.packed_cat = torch.cat((self.packed_state_cache, self.packed_state), dim=1)

        self.boxes_area = torch.zeros(batch_size, dtype=torch.float, device=device)
        self.total_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
        self.skyline = torch.zeros(batch_size, position_size, dtype=float, device=device)
        self.action=PackAction(batch_size, device=device)

    def put_reward(self, reward):
        self.total_rewards += reward

    def get_rewards(self):
        return self.total_rewards

    def get_mask(self):


        block_size = self.packed_state.size(1)
        mask_array = torch.from_numpy(np.tril(np.ones(block_size), k=-1).astype('bool_')).to(self.packed_state.device)
        
        # we have one zero block first, so we have to pack one more block
        remain_steps = self.instance_size + block_size - self.i
        assert remain_steps > 0, 'over packed!!!'
        
        
        if remain_steps // block_size == 0:
            # from one to (block_size-1)
            mask_num = block_size - remain_steps
        else:
            mask_num = 0
            
        return mask_array[mask_num]


    def update_env(self, new_box):

        # new_box (batch, 1, 2)
        if self.online:
            self.boxes = new_box
        else:
            batch_size, block_size, box_state_size = self.boxes.size()

            all_index = torch.arange(block_size, device=self.boxes.device).repeat(batch_size, 1)

            # we initialize index with 0, so it doesn't matter with the first one
            mask = (all_index!=self.action.index).unsqueeze(-1).repeat(1, 1, box_state_size)
            # selected_box (batch, block-1, box_state_size)
            remaining_boxes = torch.masked_select(self.boxes, mask).view(batch_size, -1, box_state_size)

            self.boxes = torch.cat((new_box, remaining_boxes), dim=1)


    def update_select(self, selected):
        # select(batch,1)
        self.action.set_index(selected)

        # set raw shape
        box_width, box_height = self._get_action_box_shape()
        # print("box width, box height", self.boxes[0], selected[0], box_width[0], box_height[0])
        self.action.set_shape(box_width, box_height) 


    def _get_action_box_shape(self):

        if self.online:
            box_width = self.boxes[:, :, 0].squeeze(-1).squeeze(-1) # (batch)
            box_height = self.boxes[:, :, 1].squeeze(-1).squeeze(-1)
        else:
            select_index = self.action.index.long()

            box_raw_w = self.boxes[:, :, 0].squeeze(-1) # (batch, graph)
            box_raw_h = self.boxes[:, :, 1].squeeze(-1)
            # print("box_raw_w: ", box_raw_w)
            
            # print(box_raw_w.size(), select_index.size())
            box_width = torch.gather(box_raw_w, -1, select_index).squeeze(-1) # (batch)
            box_height = torch.gather(box_raw_h, -1, select_index).squeeze(-1)

        return box_width, box_height


    # update roate action and set width and height according rotate
    def update_rotate(self, rotate):
        # rotate(batch, 1)
        
        self.action.set_rotate(rotate)
        
        rotate_mask = rotate.squeeze(-1).gt(0.5) # (batch)
        
        # (batch, 2)
        box_shape = self.action.get_shape()

        box_width = box_shape[:, 0] # (batch)
        box_height = box_shape[:, 1]

        box_width_r = box_height
        box_height_r = box_width
        
        box_width_r = torch.masked_select(box_width_r, rotate_mask) # (all rotated)
        box_height_r = torch.masked_select(box_height_r, rotate_mask)
        # print("box_width_r: ", box_width_r)

        inbox_width = box_width.masked_scatter(rotate_mask, box_width_r) # (batch)
        inbox_height = box_height.masked_scatter(rotate_mask, box_height_r)


        # save to action
        self.action.set_shape(inbox_width, inbox_height) 



    # set x,y and update packing state
    def update_pack(self, x):
        
        batch_size = self.packed_state.size(0)
        select_index = self.action.index.squeeze(-1).long()
        
        # y = self._get_y(x)
        y = self._get_y_skyline(x)
        self.action.set_pos(x, y)
        
        # (batch, 1, 4)
        packed_box = self.action.get_packed().unsqueeze(-2)

        inbox_shape = self.action.get_shape()
        # add new box area
        self.boxes_area += (inbox_shape[:,0] * inbox_shape[:,1]).squeeze(-1)

        # FIFO packed!!!
        # print('packed_box 0 and action.get_shape 0:', packed_box[0], self.action.get_shape()[0])
        # print('packed state:', self.packed_state[0])
        self.packed_state_cache = push_to_tensor_alternative(self.packed_state_cache, self.packed_state[:,0:1,:])
        self.packed_state = push_to_tensor_alternative(self.packed_state, packed_box)
        
        self.packed_cat = torch.cat((self.packed_state_cache, self.packed_state), dim=1)

        self.i += 1
    
    def _get_y_skyline(self, x):

        inbox_width = self.action.get_packed()[:,0]
        inbox_height = self.action.get_packed()[:,1].unsqueeze(-1)
        position_size = self.skyline.size(1)
        batch_size = self.skyline.size(0)

        in_left = torch.min(x.squeeze(-1), x.squeeze(-1) + inbox_width)
        in_right = torch.max(x.squeeze(-1), x.squeeze(-1) + inbox_width)

        # print(in_left, in_right)

        left_idx = ((in_left + 1.0) * (position_size/2)).floor().long().unsqueeze(-1)
        right_idx = ((in_right + 1.0) * (position_size/2)).floor().long().unsqueeze(-1)
        

        mask = torch.arange(0, position_size, device=self.device).repeat(batch_size,1)
        mask_left = mask.ge(left_idx)
        mask_right = mask.le(right_idx)
        mask = mask_left * mask_right
        masked_skyline = mask * self.skyline
        non_masked_skyline = (~mask) * self.skyline
        # print("mask skyline size:", masked_skyline.size(), non_masked_skyline.size(), masked_skyline[0])
        max_y = torch.max(masked_skyline, 1)[0].unsqueeze(-1).float()
        # print("max y size:", max_y.size(), max_y[0])
        # print("inbox height:", inbox_height.size(), inbox_height[0])
        update_skyline = mask * (max_y + inbox_height) + non_masked_skyline
        # print("000skyline 0:", self.skyline[0], left_idx[0], right_idx[0])
        self.skyline = update_skyline
        # print("111skyline 0",self.skyline[0])

        return max_y



    def _get_y(self, x):

        inbox_width = self.action.get_packed()[:,0]

        in_left = torch.min(x.squeeze(-1), x.squeeze(-1) + inbox_width)  # (batch)
        in_right = torch.max(x.squeeze(-1), x.squeeze(-1) + inbox_width)

        box_width = self.packed_cat[:, :, 0]
        box_height = self.packed_cat[:, :, 1]

        box_x = self.packed_cat[:, :, 2] # (batch, packed)
        box_y = self.packed_cat[:, :, 3]

        box_left = torch.min(box_x, box_x + box_width) # (batch, packed)
        box_right = torch.max(box_x, box_x + box_width)
        box_top = torch.max(box_y, box_y + box_height)
        
        in_left = in_left.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]]) # (batch, packed)
        in_right = in_right.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]])

        # print(box_right.size(), in_left.size())

        is_left = torch.gt(box_right, in_left)  # box_right > in_left  # (batch, packed)
        is_right = torch.lt(box_left, in_right) # box_left < in_right

        is_overlaped = is_left * is_right # element wise multiplication just logic &(and)
        # print("is_overlaped size", is_overlaped.size())
        non_overlaped = ~is_overlaped

        overlap_box_top = box_top.masked_fill(non_overlaped, 0) # (batch, select)
        # print("overlap_box_top size", overlap_box_top.size())

        max_y, _ = torch.max(overlap_box_top, -1, keepdim=True) # (batch, 1)
        
        return max_y

    def get_boundx(self):

        batch_size = self.packed_state.size()[0]
        
        right_b = torch.ones(batch_size, device=self.packed_state.device) - self.action.get_shape()[:,0]
        
        return right_b

    def get_height(self):

        box_height = self.packed_cat[:, :, 1]

        # (batch, packed)
        box_top = self.packed_cat[:, :, 3] + box_height

        heights, _ = torch.max(box_top, -1) 

        # (batch)
        return heights

    def get_gap_size(self):
        
        bin_area = self.get_height() * 2.0

        gap_area = bin_area - self.boxes_area
        
        return gap_area

    def get_gap_ratio(self):
        
        # (batch)   bin width is 2
        bin_area = self.get_height() * 2.0
        
        gap_ratio = self.get_gap_size() / bin_area
        
        return gap_ratio


    def get_graph(self):
    # we drop boxes of graph for rendering
        # drop_height = 6.0

        # graph = copy.deepcopy(self.packed_cat)

        # drop_graph = torch.ge(self.get_height(), drop_height) # (batch)

        # mask = torch.nonzero(drop_graph).squeeze(-1) # (mask)
        
        # if mask.size(0)!=0:
        #     min_y, _ = torch.min(graph[mask, :, 3], dim=1) #(mask)
        #     graph[mask, :, 3] -= min_y.unsqueeze(-1).repeat([1, graph.size()[1]])

        # graph(width, height, x, y)
        return self.packed_cat

