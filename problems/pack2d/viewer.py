#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


import pyglet
from pyglet.gl import *
from pyglet.image import ImagePattern, ImageData
from pyglet.window import key
import torch

LEFT_SPACE = 100


pyglet.resource.path = ['resources']
pyglet.resource.reindex()
pyglet.resource.add_font('MONOFONT.TTF')

BLACK = (0.2, 0.2, 0.2, 1.0)      # color 0
RED = (1.0, 0.5, 0.5, 1.0)        # color 1
GREEN = (0.5, 1.0, 0.5, 1.0)      # color 2
BLUE = (0.5, 0.5, 1.0, 1.0)       # color 3
YELLOW = (0.8, 0.8, 0.5, 1.0)     # color 4
GREY = (0.7, 0.7, 0.7, 1.0)       # color 5
DARK_GREY = (0.5, 0.5, 0.5, 1.0)  # color 6
WHITE = (0.9, 0.9, 0.9, 1.0)      # color 7

COLORS = [BLACK, RED, GREEN, BLUE, YELLOW, GREY, DARK_GREY, WHITE]

def int_color(float_color):
    return tuple(map(lambda c: int(c*255), float_color))


def lighten_color(float_color):
    return tuple(map(lambda c: min(c*1.2, 1.0), float_color))


def darken_color(float_color):
    return tuple(map(lambda c: c*0.8, float_color))

class BlockImagePattern(ImagePattern):
    def __init__(self, color):
        self.color = int_color(color)
        self.frame_color = int_color(BLACK)

    def create_image(self, width, height):
        data = b''
        for i in range(width * height):
            pos_x = i % width
            pos_y = i // width
            if pos_x < 2 or pos_x > width - 2:
                data += bytes(self.frame_color)
            elif pos_y < 2 or pos_y > height - 2:
                data += bytes(self.frame_color)
            else:
                data += bytes(self.color)
        return ImageData(width, height, 'RGBA', data)


# In[6]:


class Viewer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.bin_width = 100
        self.window = pyglet.window.Window(width, 
                                             height,
                                             resizable=True, 
                                             caption='Packing problem', 
                                             visible=True)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.scale = 1
        
        self.geoms = []
        
        self.batch = pyglet.graphics.Batch()
        
        
    def set_scale(self, scale):
        self.scale = scale
        self.bin_width = scale*2
        
    def draw_background(self, width):
        

        pyglet.gl.glClearColor(*WHITE)


        color = lighten_color(BLACK) * 2

        self.batch.add(2, pyglet.gl.GL_LINES, None,
                ('v2f', (width+LEFT_SPACE, 0,
                        width+LEFT_SPACE, self.height)),
                ('c4f', color))
        
        self.batch.add(2, pyglet.gl.GL_LINES, None,
                ('v2f', (LEFT_SPACE, 0,
                        LEFT_SPACE, self.height)),
                ('c4f', color))
    
    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        
    def draw_text(self, height, gap_size, gap_ratio):
        
        text = "height: " + str(format(height.data.cpu().numpy(), '.3f'))
        height_label = pyglet.text.Label(text=text, color=(0,0,0,255), \
                                        x=self.width*4/5, y=self.height-40, batch=self.batch)
        
        text = "gap_size: " + str(format(gap_size.data.cpu().numpy(), '.3f'))
        height_label = pyglet.text.Label(text=text, color=(0,0,0,255), \
                                        x=self.width*4/5, y=self.height-60, batch=self.batch)
        
        text = "gap_ratio: " + str(format(gap_ratio.data.cpu().numpy(), '.3f'))
        height_label = pyglet.text.Label(text=text, color=(0,0,0,255), \
                                        x=self.width*4/5, y=self.height-80, batch=self.batch)
        
        
    def draw_top_line(self, height):
        
        g_height = height * self.scale
        
        black_color = lighten_color(BLACK) * 2
        self.batch.add(2, pyglet.gl.GL_LINES, None,
                ('v2f', (LEFT_SPACE, g_height,
                        LEFT_SPACE+self.bin_width, g_height)),
                ('c4f', black_color))
        
        
    def add_geom(self, box, i):
        
        # (graph)
        
        # width must be positive, so we need rotate back
        rotate = box[0].lt(0)
        
        color = darken_color(BLUE) 

        box_width = box[0]
        box_height = box[1]

        box_x = (box[2] + 1) # (0~2)
        box_y = box[3]

        box_image = pyglet.image.create(int(box_width*self.scale), int(box_height*self.scale),
                                BlockImagePattern(color))
        box_obj = pyglet.sprite.Sprite(box_image, LEFT_SPACE+box_x*self.scale, box_y*self.scale, batch=self.batch)
        box_obj.rotation = 0 

        self.geoms.append(box_obj)

        center_x = (box_x + box[0] /2) * self.scale + LEFT_SPACE
        center_y = (box_y + box[1] / 2) * self.scale

        box_order_label = pyglet.text.Label(text=str(i), color=(0, 0, 0, 255), x=center_x, y=center_y, batch=self.batch)


    def render(self):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.batch.draw()
        self.window.flip()
        # self.geoms = []
        
    def __del__(self):
        self.close()




