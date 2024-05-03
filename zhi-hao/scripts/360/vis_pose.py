import json
import os
from argparse import ArgumentParser

import numpy as np
import vedo

parser = ArgumentParser()
parser.add_argument('transform', help='Path to transform json')
args = parser.parse_args()

path_src = args.transform
with open(path_src, 'r') as f:
    data = json.load(f)

frames = data['frames']
c2ws = np.array([frames[i]['transform_matrix'] for i in range(len(frames))])
c2ws = c2ws[:, :3, :]

# c2ws[:, :, -1] *= 10
pos = c2ws[:, :, -1]
arrow_len, s = 0.1, 1
x_end   = pos + arrow_len * c2ws[:, :, 0]
y_end   = pos + arrow_len * c2ws[:, :, 1]
z_end   = pos + arrow_len * c2ws[:, :, 2]

x = vedo.Arrows(pos, x_end, s=s, c='red')
y = vedo.Arrows(pos, y_end, s=s, c='green')
z = vedo.Arrows(pos, z_end, s=s, c='blue')
    
vedo.show(x,y,z, axes=1)