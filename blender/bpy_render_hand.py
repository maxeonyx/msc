import bpy
from bpy import data as D
from bpy import context as C
import sys
import os
import datetime
from dotmap import DotMap as dm

file_dir = os.path.dirname(bpy.data.filepath)
if not file_dir in sys.path:
    sys.path.append(file_dir)

import bpy_animate_addon

def select_hand(cfg):
    C.scene.objects['leftHand'].select_set(True)
    C.scene.objects['rightHand'].select_set(False)

def animate(cfg):    
    bpy.ops.object.generate_animation(conditioning_steps=cfg.conditioning_steps, new_frames=cfg.new_frames, window_size=30)

def render(cfg):

    bpy.context.scene.render.resolution_x = 1600
    bpy.context.scene.render.resolution_y = 900
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = cfg.conditioning_steps + cfg.new_frames
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    bpy.context.scene.render.filepath = "//renders/" + timestamp
    bpy.context.scene.render.fps = 60

    if cfg.cuda:

        bpy.context.scene.render.engine = 'CYCLES'
        # bpy.context.scene.cycles.max_bounces = 6
        bpy.context.scene.cycles.samples = 256
        
        # Set the device_type
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"

        # Set the device and feature set
        bpy.context.scene.cycles.device = "GPU"

        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1 # Using all devices, include GPU and CPU
            print(d["name"], d["use"])
        
        bpy.ops.render.render(animation=True)
    else:
        bpy.ops.render.opengl(animation=True)

    
if __name__ == '__main__':
    cfg = dm(
        conditioning_steps = 1,
        new_frames = 500,
        cuda = True,
    )
    select_hand(cfg)
    animate(cfg)
    render(cfg)
