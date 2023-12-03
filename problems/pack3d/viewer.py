#!/usr/bin/env python
# coding: utf-8


from __future__ import division
from vpython import *
# from visual import *
import torch
import random
import numpy as np

scene.caption = """Right button drag or Ctrl-drag to rotate "camera" to view scene.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
    On a two-button mouse, middle is left + right.
Shift-drag to pan left/right and up/down.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate."""
scene.width = 800
scene.height = 800

BLACK = vector(0, 0, 0)           # color 0
BLUE = vector(0, 0, 1)            # color 1
GREEN = vector(0, 1, 0)           # color 2
CYAN = vector(0, 1, 1)            # color 3
RED = vector(1, 0, 0)             # color 4
MAGENTA = vector(1, 0, 1)         # color 5
YELLOW = vector(1, 1, 0)          # color 6
WHITE = vector(1, 1, 1)           # color 7
GRAY = vector(0.5, 0.5, 0.5)      # color 8

COLORS = [GRAY, BLUE, GREEN, CYAN, RED, MAGENTA, YELLOW, WHITE, BLACK]



class Viewer(object):
    def __init__(self, width, height, length):
        self.width= width
        self.height = height
        self.length = length
        self.thick = 0.001
        wallR = box(pos=vector(self.width/2, self.height/2-self.height/2, 0), size=vector(self.thick, self.height, self.length), color=COLORS[7], opacity=0.2)
        wallL = box(pos=vector(-self.width/2, self.height/2-self.height/2, 0), size=vector(-self.thick, self.height, self.length), color=COLORS[7], opacity=0.2)
        wallF = box(pos=vector(0, self.height/2-self.height/2, self.length/2), size=vector(self.width, self.height, self.thick), color=COLORS[7], opacity=0.2)
        wallB = box(pos=vector(0, self.height/2-self.height/2, -self.length/2), size=vector(self.width, self.height, -self.thick), color=COLORS[7], opacity=0.2)
        wallD = box(pos=vector(0, 0-self.height/2, 0), size=vector(self.width, self.thick, self.length), color=COLORS[7], opacity=0.4)


    def add_geom(self, boxs):
        # color_index = random.randint(1,6)
        boxs = boxs.tolist()
        # print(boxs[6])
        pos = vector(boxs[4] + boxs[1]/2, boxs[5] + boxs[2]/2 - self.height/2, boxs[3] + boxs[0]/2)
        size = vector(boxs[1], boxs[2], boxs[0])
        # print(size)
        # color_index = int(boxs[6])


        add_box = box(pos=pos, size=size, color=random.choice(COLORS), opacity=0.9)



# viewer = Viewer(2, 2, 2)

