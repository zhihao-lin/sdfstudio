import json
import os
import pickle

import bpy
import numpy as np
from mathutils import Matrix


class Camera:
    def __init__(self, cam_path, out_dir, format="sdfstudio"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.format = format

        transform = None
        with open(cam_path, "rb") as file:
            transform = json.load(file)
        if format == "sdfstudio":
            w, h = transform["width"], transform["height"]
            frames = transform["frames"]
            paths = [frame["rgb_path"] for frame in frames]
            sorted_idx = np.argsort(paths)
            c2w = np.array([frame["camtoworld"] for frame in frames])[sorted_idx]  # OpenCV
            c2w[:, :3, 1:3] *= -1  # to Blender/OpenGL coordinate system
            Ks = np.array([frame["intrinsics"] for frame in frames])[sorted_idx]
            K = np.mean(Ks, axis=0)[:3, :3]
        else:
            w, h = transform["w"], transform["h"]
            fl_x, fl_y = transform["fl_x"], transform["fl_y"]
            cx, cy = transform["cx"], transform["cy"]
            K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
            frames = transform["frames"]
            paths = [frame["file_path"] for frame in frames]
            sorted_idx = np.argsort(paths)
            c2w = np.array([frame["transform_matrix"] for frame in frames])[sorted_idx]

        # Set intrinsics
        bpy.data.scenes["Scene"].render.resolution_x = w
        bpy.data.scenes["Scene"].render.resolution_y = h
        bpy.data.scenes["Scene"].render.resolution_percentage = 100

        bpy.data.cameras["Camera"].type = "PERSP"
        bpy.data.cameras["Camera"].sensor_fit = "HORIZONTAL"
        bpy.data.cameras["Camera"].lens_unit = "FOV"
        fx = K[0, 0]
        rad = 2 * np.arctan(w / (2 * fx))
        bpy.data.cameras["Camera"].angle = rad
        cx = K[0, 2]
        cy = K[1, 2]
        bpy.data.cameras["Camera"].shift_x = -(cx - w / 2) / w
        bpy.data.cameras["Camera"].shift_y = (cy - h / 2) / h
        # bpy.data.objects['Camera'].set_intrinsics_from_K_matrix(K, w, h)

        self.camera = bpy.data.objects["Camera"]
        self.poses = c2w
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.samples = 1024
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True
        # Set the device_type
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"

        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1  # Using all devices, include GPU and CPU
            print(d["name"], d["use"])

    def move_to_frame(self, index):
        print(self.poses[index])
        self.camera.matrix_world = Matrix(self.poses[index])

    def render_rgb(self, index, dir_name="rgb"):
        dir_path = os.path.join(self.out_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        # num = self.poses.shape[0]
        self.move_to_frame(index)
        img_path = os.path.join(dir_path, "{:0>3d}.png".format(index))
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.render(use_viewport=True, write_still=True)

    def initialize_depth_extractor(self):
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        bpy.context.scene.use_nodes = True

        node_tree = bpy.data.scenes["Scene"].node_tree
        render_layers = node_tree.nodes["Render Layers"]
        node_tree.nodes.new(type="CompositorNodeOutputFile")
        file_output = node_tree.nodes["File Output"]
        file_output.format.file_format = "OPEN_EXR"
        links = node_tree.links
        new_link = links.new(render_layers.outputs[2], file_output.inputs[0])

    def render_depth(self, index, dir_name="depth"):
        self.initialize_depth_extractor()
        dir_path = os.path.join(self.out_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        # num = self.poses.shape[0]
        self.move_to_frame(index)
        bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path = os.path.join(
            dir_path, "{:0>3d}".format(index)
        )
        bpy.ops.render.render(use_viewport=True, write_still=True)

    # def render_rgb_and_depth(self, index, dir_name_rgb='rgb', dir_name_depth='depth'):
    #     self.initialize_depth_extractor()  # Assuming this is needed for depth rendering setup

    #     dir_path_rgb = os.path.join(self.out_dir, dir_name_rgb)
    #     dir_path_depth = os.path.join(self.out_dir, dir_name_depth)
    #     os.makedirs(dir_path_rgb, exist_ok=True)
    #     os.makedirs(dir_path_depth, exist_ok=True)

    #     self.move_to_frame(index)

    #     # Set paths for both RGB and depth outputs
    #     depth_output_path = os.path.join(dir_path_depth, '{:0>3d}'.format(index))
    #     rgb_output_path = os.path.join(dir_path_rgb, '{:0>3d}.png'.format(index))

    #     # Assuming your Blender setup has nodes named accordingly
    #     bpy.context.scene.render.filepath = rgb_output_path
    #     bpy.data.scenes["Scene"].node_tree.nodes["File Output Depth"].base_path = depth_output_path
    #     bpy.ops.render.render(use_viewport=True, write_still=True)


if __name__ == "__main__":
    pass
