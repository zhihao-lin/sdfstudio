import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def move_images():
    src_0 = "normal_00"
    src_1 = "normal_01"
    tgt = "normals"
    os.makedirs(tgt, exist_ok=True)

    img_paths = sorted([os.path.join(src_0, name) for name in os.listdir(src_0)])
    img_paths += sorted([os.path.join(src_1, name) for name in os.listdir(src_1)])
    for i in tqdm(range(len(img_paths))):
        img = Image.open(img_paths[i])
        out_path = os.path.join(tgt, "{:0>4d}_normal.png".format(i))
        img.save(out_path)


def process_normal():
    dir_normal = "normals"
    images = sorted([os.path.join(dir_normal, name) for name in os.listdir(dir_normal) if name.endswith("png")])
    for i in tqdm(range(len(images))):
        img = np.array(Image.open(images[i]))
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        path = os.path.join(dir_normal, "{:0>4d}_normal.npy".format(i))
        np.save(path, img)


def parse_calib_file(path, key):
    file = open(path, "r")
    lines = file.readlines()
    for line in lines:
        if key in line:
            tokens = line.strip().split(" ")
            nums = tokens[1:]
            array = np.array([float(i) for i in nums])
            return array
    return None


def generate_sdfstudio():
    frame_start = 1538
    frame_end = 1601
    dir_calib = "kitti360_test/calibration"
    dir_pose = "kitti360_test/data_poses"
    seq = "2013_05_28_drive_0000_sync"
    output_path = "kitti360-1538-1601/meta_data.json"
    scene_scale = 2

    # intrinsics
    intrinsic_path = os.path.join(dir_calib, "perspective.txt")
    K = parse_calib_file(intrinsic_path, "P_rect_00").reshape(3, 4)
    K = np.concatenate([K, np.array([[0, 0, 0, 1]])], axis=0)
    img_size = parse_calib_file(intrinsic_path, "S_rect_00")
    w, h = int(img_size[0]), int(img_size[1])

    # Extrinsics
    dir_poses = os.path.join(dir_pose, seq)
    pose_cam_0 = np.genfromtxt(os.path.join(dir_poses, "cam0_to_world.txt"))  # (n, 17)
    frame_id = pose_cam_0[:, 0]
    sample = np.logical_and(frame_id >= frame_start, frame_id <= frame_end)

    cam2world_0 = pose_cam_0[sample, 1:].reshape(-1, 4, 4)[:, :3]
    sys2world = np.genfromtxt(os.path.join(dir_poses, "poses.txt"))
    sys2world = sys2world[sample, 1:].reshape(-1, 3, 4)
    cam2sys_1 = parse_calib_file(os.path.join(dir_calib, "calib_cam_to_pose.txt"), "image_01")
    cam2sys_1 = np.concatenate([cam2sys_1.reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
    R_rect_01 = parse_calib_file(intrinsic_path, "R_rect_01").reshape(3, 3)
    R_rect = np.eye(4)
    R_rect[:3:, :3] = np.linalg.inv(R_rect_01)
    cam2world_1 = sys2world @ cam2sys_1 @ R_rect
    cam2world = np.concatenate([cam2world_0, cam2world_1], axis=0)
    # normalize to cube (-1, -1, -1)~(1, 1, 1)
    pos = cam2world[:, :, -1]  # (n, 3)
    forward = pos[-1] - pos[0]
    xyz_min = pos.min(axis=0)
    xyz_max = pos.max(axis=0)
    center = (xyz_max + xyz_min) / 2.0
    scale = (xyz_max - xyz_min).max() / 2.0
    pos = (pos - center[None]) / scale
    # pos = pos - forward[None] * 0.5
    cam2world[:, :, -1] = pos
    ones = np.zeros((len(cam2world), 1, 4))
    ones[:, :, -1] = 1
    cam2world = np.concatenate([cam2world, ones], axis=1)
    aabb = np.array([[-1, -1, -1], [1, 1, 1]]) * scene_scale

    # Export meta data
    meta_data = {
        "camera_model": "OPENCV",
        "height": h,
        "width": w,
        "has_mono_prior": True,
        "pairs": None,
        "worldtogt": np.eye(4).tolist(),
        "scene_box": {
            "aabb": aabb.tolist(),
            "near": 0.01,
            "far": 2.5 * scene_scale,
            "radius": scene_scale,
            "collider_type": "box",
        },
        "frames": [],
    }
    for i in range(len(cam2world)):
        frame_data = {
            "rgb_path": "{:0>4d}_rgb.png".format(i),
            "camtoworld": cam2world[i].tolist(),
            "intrinsics": K.tolist(),
            "mono_depth_path": "{:0>4d}_depth.npy".format(i),
            "mono_normal_path": "{:0>4d}_normal.npy".format(i),
        }
        meta_data["frames"].append(frame_data)

    with open(output_path, "w") as file:
        json.dump(meta_data, file, indent=4)


if __name__ == "__main__":
    # process_normal()
    generate_sdfstudio()
