import sys

import bpy
import numpy as np

ROOT = "/hdd/sdfstudio"
sys.path.append(ROOT)
BLENDER = "/hdd/sdfstudio/zhi-hao/blender"
sys.path.append(BLENDER)
import imp
import os

import camera

imp.reload(camera)


cam_path = os.path.join(ROOT, "data/colmap_sdfstudio/garden/transforms.json")
out_dir = ROOT
format = "nerfstudio"
frame_idx = 0
cam = camera.Camera(cam_path, out_dir, format)
cam.move_to_frame(frame_idx)
