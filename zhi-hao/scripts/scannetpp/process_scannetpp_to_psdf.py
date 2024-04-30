import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--id", type=str, default="4a1a3a7dc5")
args = parser.parse_args()

data_dir = f"/hdd/datasets/scannetpp/data/{args.id}/dslr/"
save_dir = f"/hdd/datasets/scannetpp/data/{args.id}/psdf/"

os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
camera_path = os.path.join(data_dir, "colmap", "cameras.txt")
split_json = os.path.join(data_dir, "train_test_lists.json")

with open(os.path.join(data_dir, "nerfstudio", "transforms.json"), "r") as f:
    original_ns_json = json.load(f)

with open(split_json, "r") as f:
    splits = json.load(f)

with open(camera_path, "r") as f:
    f.readline()
    f.readline()
    f.readline()
    info = f.readline()
    info = info.split(" ")
    info = [elem for elem in info if len(info) > 0]
    info = info[2:]

W = int(info[0])
H = int(info[1])
DIM = (W, H)

MAX_W = 1024

fx = float(info[2])
fy = float(info[3])
cx = float(info[4])
cy = float(info[5])

K = np.array(
    [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
)

D = np.array(
    [
        float(info[6]),
        float(info[7]),
        float(info[8]),
        float(info[9]),
    ],
)

newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), 1)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newcameramtx, DIM, cv2.CV_32FC1)

ns_json = {}

ns_json["w"] = int(MAX_W)
ns_json["h"] = int(H * (MAX_W / W))

ns_json["fl_x"] = float(newcameramtx[0, 0]) * (ns_json["w"] / W)
ns_json["fl_y"] = float(newcameramtx[1, 1]) * (ns_json["h"] / H)

ns_json["cx"] = float(newcameramtx[0, 2]) * (ns_json["w"] / W)
ns_json["cy"] = float(newcameramtx[1, 2]) * (ns_json["h"] / H)

ns_json["camera_model"] = "OPENCV"
ns_json["frames"] = []

paths = []
transforms = []
for frame in tqdm(original_ns_json["frames"]):
    name = frame["file_path"]
    fisheye_path = os.path.join(data_dir, "resized_images", name)
    save_path = os.path.join(save_dir, "images", name)
    img = cv2.imread(fisheye_path)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    undistorted_img = Image.fromarray(undistorted_img)
    undistorted_img = undistorted_img.resize((int(MAX_W), int(H * (MAX_W / W))))
    undistorted_img.save(save_path)

    # ns_json["frames"].append({
    #     "file_path": os.path.join("images", name),
    #     "transform_matrix": frame["transform_matrix"]
    # })
    paths.append(os.path.join("images", name))
    transforms.append(frame["transform_matrix"])

transforms = np.array(transforms)
ts = transforms[:, :3, -1]
min_vertices = ts.min(axis=0)
max_vertices = ts.max(axis=0)
center = (min_vertices + max_vertices) / 2
scale = 2.0 / np.max(max_vertices - min_vertices)
ts = (ts - center) * scale
transforms[:, :3, -1] = ts

for i in tqdm(range(len(paths))):
    f = {"file_path": paths[i], "transform_matrix": transforms[i].tolist()}
    ns_json["frames"].append(f)

with open(os.path.join(save_dir, "transforms.json"), "w") as f:
    json.dump(ns_json, f, indent=4)

test_paths = []
test_transforms = []
for frame in tqdm(original_ns_json["test_frames"]):
    name = frame["file_path"]
    fisheye_path = os.path.join(data_dir, "resized_images", name)
    save_path = os.path.join(save_dir, "images", name)
    img = cv2.imread(fisheye_path)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    undistorted_img = Image.fromarray(undistorted_img)
    undistorted_img = undistorted_img.resize((int(MAX_W), int(H * (MAX_W / W))))
    undistorted_img.save(save_path)

    test_paths.append(os.path.join("images", name))
    test_transforms.append(frame["transform_matrix"])

test_transforms = np.array(test_transforms)
ts = test_transforms[:, :3, -1]
ts = (ts - center) * scale
test_transforms[:, :3, -1] = ts

for i in tqdm(range(len(test_paths))):
    f = {"file_path": test_paths[i], "transform_matrix": test_transforms[i].tolist()}
    ns_json["frames"].append(f)

with open(os.path.join(save_dir, "transforms_all.json"), "w") as f:
    json.dump(ns_json, f, indent=4)