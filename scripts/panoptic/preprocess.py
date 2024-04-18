import torch
import torchvision
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import gzip
from pathlib import Path
import json
import pickle

def get_thing_semantics(sc_classes='extended'):
    thing_semantics = [False]
    for cllist in [x.strip().split(',') for x in Path(f"resources/scannet_{sc_classes}_things.csv").read_text().strip().splitlines()]:
        thing_semantics.append(bool(int(cllist[1])))
    return thing_semantics


def create_segmentation_data(sc_classes='extended'):
    thing_semantics = get_thing_semantics(sc_classes)
    print('len thing_semantics', len(thing_semantics))
    export_dict = {
        'num_semantic_classes': len(thing_semantics),
        'fg_classes': [i for i, is_thing in enumerate(thing_semantics) if is_thing],
        'bg_classes': [i for i, is_thing in enumerate(thing_semantics) if not is_thing]
    }
    return export_dict

def convert_from_mask_to_semantics_and_instances_no_remap(original_mask, segments, _coco_to_scannet, is_thing, instance_ctr, instance_to_semantic):
    id_to_class = torch.zeros(1024).int()
    instance_mask = torch.zeros_like(original_mask)
    invalid_mask = original_mask == 0
    for s in segments:
        id_to_class[s['id']] = s['category_id']
        if is_thing[s['category_id']]:
            instance_mask[original_mask == s['id']] = instance_ctr
            instance_to_semantic[instance_ctr] = s['category_id']
            instance_ctr += 1
    return id_to_class[original_mask.flatten().numpy().tolist()].reshape(original_mask.shape), instance_mask, invalid_mask, instance_ctr, instance_to_semantic

sc_classes = "extended"
segment_data = create_segmentation_data(sc_classes)

coco_to_scannet = {}
thing_semantics = get_thing_semantics(sc_classes)
for cidx, cllist in enumerate([x.strip().split(',') for x in
                               Path(f"resources/scannet_{sc_classes}_to_coco.csv").read_text().strip().splitlines()]):
    for c in cllist[1:]:
        coco_to_scannet[c.split('/')[1]] = cidx + 1
instance_ctr = 1
instance_to_semantic = {}
segment_ctr = 1

data_dir = "/home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/"

with open(os.path.join(data_dir, "psdf", "transforms.json"), 'r') as f:
    ns_json = json.load(f)

MAX_RESOLUTION = 512

# image_names = sorted(os.listdir(os.path.join(data_dir, "perspective")))

os.makedirs(os.path.join(data_dir, "plift", "images"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "plift", "segments"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "plift", "semantics"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "plift", "instance"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "plift", "invalid"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "plift", "probabilities"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "plift", "confidences"), exist_ok=True)

for frame_i, frame in tqdm(enumerate(ns_json["frames"]), total=len(ns_json["frames"])):
    name = os.path.basename(frame["file_path"])
    image_path = os.path.join(os.path.join(data_dir, "perspective", name))
    image = torch.from_numpy(np.array(Image.open(image_path)) / 255.0)
    H, W = image.shape[:2]

    ratio = max(H, W) / MAX_RESOLUTION
    h, w = int(H / ratio), int(W / ratio)

    image = torchvision.transforms.functional.resize(image.permute(2, 0, 1), (h, w)).permute(1, 2, 0)
    image = Image.fromarray((image * 255.0).numpy().clip(0, 255).astype(np.uint8))
    image.save(os.path.join(data_dir, "plift", "images", name))

    ns_json["frames"][frame_i]["file_path"] = os.path.join(data_dir, "plift", "images", name)
    if frame_i == 0:
        ns_json["fl_x"] = ns_json["fl_x"] / ratio
        ns_json["fl_y"] = ns_json["fl_y"] / ratio
        ns_json["cx"] = ns_json["cx"] / ratio
        ns_json["cy"] = ns_json["cy"] / ratio
        ns_json["w"] = w
        ns_json["h"] = h

    stem = os.path.splitext(name)[0]
    panoptic_path = os.path.join(data_dir, "panoptic", stem+".ptz")

    data = torch.load(gzip.open(panoptic_path), map_location='cpu')
    probability, confidence = data['probabilities'], data['confidences']

    semantic, instance, invalid_mask, instance_ctr, instance_to_semantic = convert_from_mask_to_semantics_and_instances_no_remap(
        data['mask'], data['segments'], coco_to_scannet, thing_semantics, instance_ctr,
        instance_to_semantic)

    segment_mask = torch.zeros_like(data['mask'])
    for s in data['segments']:
        segment_mask[data['mask'] == s['id']] = segment_ctr
        segment_ctr += 1

    segment_mask = torchvision.transforms.functional.resize(segment_mask.unsqueeze(0).long(), (h, w),
                                                            torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
    semantic = torchvision.transforms.functional.resize(semantic.unsqueeze(0).long(), (h, w),
                                                        torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
    instance = torchvision.transforms.functional.resize(instance.unsqueeze(0).long(), (h, w),
                                                        torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
    invalid_mask = torchvision.transforms.functional.resize(invalid_mask.unsqueeze(0).bool(), (h, w),
                                                            torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)

    pc = torch.cat([probability, confidence.unsqueeze(-1)], dim=-1).permute(2, 0, 1)
    pc = torchvision.transforms.functional.resize(pc, (h, w),
                                                  torchvision.transforms.InterpolationMode.BILINEAR).permute(1, 2, 0)
    probability = pc[..., :-1]
    confidence = pc[..., -1]

    np.save(os.path.join(data_dir, "plift", "segments", stem+".npy"), segment_mask.numpy().astype(np.uint16))
    np.save(os.path.join(data_dir, "plift", "semantics", stem+".npy"), semantic.numpy().astype(np.uint16))
    np.save(os.path.join(data_dir, "plift", "instance", stem+".npy"), instance.numpy())
    np.save(os.path.join(data_dir, "plift", "invalid", stem+".npy"), invalid_mask.numpy())
    np.save(os.path.join(data_dir, "plift", "probabilities", stem+".npy"), probability.numpy())
    np.save(os.path.join(data_dir, "plift", "confidences", stem+".npy"), confidence.numpy())

segment_data["instance_to_semantic"] = instance_to_semantic

with open(os.path.join(data_dir, "plift", "transforms.json"), 'w') as f:
    json.dump(ns_json, f, indent=4)

with open(os.path.join(data_dir, "plift", 'segmentation_data.pkl'), 'wb') as f:
    pickle.dump(segment_data, f)