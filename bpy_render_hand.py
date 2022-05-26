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

def animate(cfg):
    bpy_animate_addon.register()

    C.scene.objects['leftHand'].select_set(False)
    C.scene.objects['rightHand'].select_set(True)
    
    bpy.ops.object.generate_animation(conditioning_steps=cfg.conditioning_steps, new_frames=cfg.new_frames, window_size=30)

def render(cfg):
    
    bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.render.fps = 30
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = 480
    bpy.context.scene.render.resolution_y = 270
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = cfg.conditioning_steps + cfg.new_frames
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    bpy.context.scene.render.filepath = "//renders/" + timestamp
    
    bpy.ops.render.render(animation=True)


    
if __name__ == '__main__':
    cfg = dm(
        conditioning_steps = 1,
        new_frames = 10,
    )
    animate(cfg)
    render(cfg)