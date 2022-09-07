bl_info = {
    "name": "Generate hand animation",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import numpy as np
import json

USEFUL_COLUMNS = set([
    3, 4, 5, # Wrist

    8,  # THUMB_CMC_FE  Z-rot
    11, # THUMB_CMC_AA  Z-rot
    14, # THUMB_MCP_FE  Z-rot
    17, # THUMB_IP_FE   Z-rot

    18, # INDEX_MCP     X-rot
    20, # INDEX_MCP     Z-rot
    23, # INDEX_PIP_FE  Z-rot
    26, # INDEX_DIP_FE  Z-rot

    27, # MIDDLE_MCP    X-rot
    29, # MIDDLE_MCP    Z-rot
    32, # MIDDLE_PIP_FE Z-rot
    35, # MIDDLE_DIP_FE Z-rot

    36, # RING_MCP      X-rot
    38, # RING_MCP      Z-rot
    41, # RING_PIP_FE   Z-rot
    44, # RING_DIP_FE   Z-rot

    45, # PINKY_MCP     X-rot
    47, # PINKY_MCP     Z-rot
    50, # PINKY_PIP_FE  Z-rot
    53, # PINKY_DIP_FE  Z-rot
])

# Maps columns in the BVH data file to blender fcurve channels
# TODO: these columns aren't the best selection
COLUMN_ACCESS = {
    0: ('WRIST', 'rotation_euler', 0),
    1: ('WRIST', 'rotation_euler', 1),
    2: ('WRIST', 'rotation_euler', 2),
    3: ('THUMB_CMC_FE', 'rotation_euler', 2),
    4: ('THUMB_CMC_AA', 'rotation_euler', 2),
    5: ('THUMB_MCP_FE', 'rotation_euler', 2),
    6: ('THUMB_IP_FE', 'rotation_euler', 2),
    7: ('INDEX_MCP', 'rotation_euler', 0),
    8: ('INDEX_MCP', 'rotation_euler', 2),
    9: ('INDEX_PIP_FE', 'rotation_euler', 2),
    10: ('INDEX_DIP_FE', 'rotation_euler', 2),
    11: ('MIDDLE_MCP', 'rotation_euler', 0),
    12: ('MIDDLE_MCP', 'rotation_euler', 2),
    13: ('MIDDLE_PIP_FE', 'rotation_euler', 2),
    14: ('MIDDLE_DIP_FE', 'rotation_euler', 2),
    15: ('RING_MCP', 'rotation_euler', 0),
    16: ('RING_MCP', 'rotation_euler', 2),
    17: ('RING_PIP_FE', 'rotation_euler', 2),
    18: ('RING_DIP_FE', 'rotation_euler', 2),
    19: ('PINKY_MCP', 'rotation_euler', 0),
    20: ('PINKY_MCP', 'rotation_euler', 2),
    21: ('PINKY_PIP_FE', 'rotation_euler', 2),
    22: ('PINKY_DIP_FE', 'rotation_euler', 2),
}


class VIEW3D_PT_generate_animation(bpy.types.Panel):
    """Generate Animation Panel"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "VIEW3D_PT_generate_animation"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Generate hand animation"         # Display name in the interface.
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    
    def __init__(self):
        super().__init__()
 
    def draw(self, context):
        self.layout.label(text="Select a non-generated hand to use as conditioning")
        self.layout.label(text="(Must have 54 animation channels)")
        self.layout.label(text="Click Generate")
        self.layout.operator(GenerateAnimation.bl_idname, icon='MESH_CUBE', text="Generate")

anim_path = "//_anims/ds_bottle1_body1_left_1500.npy"

class GenerateAnimation(bpy.types.Operator):
    """Generate Animation Script"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.generate_animation"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Generate hand animation"         # Display name in the interface.
    bl_options = {'REGISTER'} 
        
    conditioning_steps: bpy.props.IntProperty(name="# Conditioning Frames", default=30, min=1, max=100)
    new_frames: bpy.props.IntProperty(name="# Generated Frames", default=100, min=1)
    window_size: bpy.props.IntProperty(name="# Frames in Autoregressive Window", default=30, min=1, max=100)
    anim_path: bpy.props.StringProperty(name="File path for animation data", default=anim_path)
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):        # execute() is called when running the operator.
        global anim_path
        wm = context.window_manager
        obj = context.object
        
        abs_anim_path = bpy.path.abspath(self.anim_path)
        meta, data = np.load(abs_anim_path, allow_pickle=True)
        
        print(json.dumps(meta))
        
        n_frames = data.shape[0]
        n_tracks = data.shape[1]
        
        if meta.type == 'dataset':
            # turn hand blue
            pass
        elif meta.type == 'generated':
            # turn hand red
            pass
        
        wm.progress_begin(0, n_frames*n_tracks)
        
        # attach a new Action to the hand
        obj.animation_data.action = bpy.data.actions.new(name="GeneratedAction")
        
        for i_track in range(n_tracks):
            track_name, track_type, track_index = COLUMN_ACCESS[i_track]
            data_path = f'pose.bones["{track_name}"].{track_type}'
            fc = obj.animation_data.action.fcurves.new(data_path=data_path, index=track_index)
            fc.keyframe_points.add(n_frames)
            for i_frame in range(n_frames):
                fc.keyframe_points[i_frame].co = (i_frame + 1.0, data[i_frame, i_track])
                wm.progress_update(i_track*n_frames+i_frame)
        
        wm.progress_end()

        return {'FINISHED'}

def menu_func(self, context):
    self.layout.operator(GenerateAnimation.bl_idname)

def register():
    bpy.utils.register_class(VIEW3D_PT_generate_animation)
    bpy.utils.register_class(GenerateAnimation)

def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_generate_animation)
    bpy.utils.unregister_class(GenerateAnimation)
