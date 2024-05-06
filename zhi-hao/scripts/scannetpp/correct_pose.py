import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import trimesh

# processed: bathroom, bathroom2, engine, game, lab, lab2, office

def main():
    parser = ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    args = parser.parse_args()

    dir_scene = os.path.join('/hdd/text-driven-vfx/datasets/scannetpp', args.scene)
    path_mesh = os.path.join(dir_scene, 'mesh', 'mesh.obj')
    path_data = os.path.join(dir_scene, 'transforms.json')

    
    with open(path_data, 'r') as f:
        data = json.load(f)
    
    transform = np.diag([-1, 1, -1, 1])
    frames = data['frames']
    for frame in frames:
        c2w = np.array(frame['transform_matrix'])
        c2w = transform @ c2w
        frame['transform_matrix'] = c2w.tolist()
    data['frames'] = frames
    
    with open(path_data, 'w') as f:
        json.dump(data, f, indent=4)
    
    mesh = trimesh.load(path_mesh)
    v = mesh.vertices
    transform = np.diag([-1, 1, -1])
    v = v @ transform.T
    mesh.vertices = v
    mesh.export(path_mesh)

def test():
    mesh_path_0 = '/hdd/text-driven-vfx/datasets/scannetpp/bathroom/mesh/mesh.obj'
    mesh_path_1 = '/hdd/text-driven-vfx/datasets/scannetpp/bathroom/mesh_flip/mesh.obj'

    mesh_0 = trimesh.load(mesh_path_0)
    v0 = mesh_0.vertices
    print(v0[:5])
    v0[:, -1] *= -1
    print(v0[:5])
    mesh_0.vertices = v0
    mesh_0.export(mesh_path_1)

    mesh_1 = trimesh.load(mesh_path_1)
    v1 = mesh_1.vertices
    print(v1[:5])

def test2():
    import cv2
    v = np.array([0, 0, 1])

    normal_temp = np.array([0.001, 0, -1])
    R_all = np.eye(3)
    for i in range(100):
        axis = np.cross(normal_temp, v)
        angle = np.arccos(np.dot(normal_temp, v))
        R = cv2.Rodrigues(axis * angle)[0]
        normal_temp = R @ normal_temp
        R_all = R @ R_all

    print(R_all)

if __name__ == '__main__':
    main()