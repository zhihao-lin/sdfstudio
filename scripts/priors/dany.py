import torchvision.transforms
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import cv2
import torch
from torchvision.transforms import Compose
import json, argparse
from tqdm import tqdm
import numpy as np
import os

encoder = 'vitl' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval().cuda()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
# /home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/perspective/transforms.json

parser = argparse.ArgumentParser()
parser.add_argument('--src', default=None, type=str, help='{samples}')
args = parser.parse_args()

with open(args.src, 'r') as f:
    info = json.load(f)

with torch.no_grad():
    for frame in tqdm(info["frames"]):
        # for img_path in img_paths:
        img_path = frame["file_path"]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
        H, W, _ = image.shape
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).cuda()
        ext = os.path.splitext(img_path)[1]

        # depth shape: 1xHxW
        depth = depth_anything(image)
        depth = torchvision.transforms.functional.resize(depth, (H, W)).squeeze(0)
        depth_max = depth.max()
        depth_min = depth.min()

        depth = 1 / (depth + 1e-4)
        depth = depth.cpu().numpy()

        target_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "depth")
        target_visdir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "depth_vis")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(target_visdir, exist_ok=True)
        target_path = os.path.join(target_dir, os.path.basename(img_path).replace(ext, '.npy'))
        target_vispath = os.path.join(target_visdir, os.path.basename(img_path).replace(ext, '.png'))

        np.save(target_path, depth)

        depth *= 10
        output = np.clip(depth, 0, 1)
        output = output * 255
        output = output.astype(np.uint8)
        output = cv2.applyColorMap(output, cv2.COLORMAP_PLASMA)

        cv2.imwrite(target_vispath, output)
