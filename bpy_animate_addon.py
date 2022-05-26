bl_info = {
    "name": "Generate hand animation",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import numpy as np




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

model_path = "//models/cuda10-hands-gpt-1layer-contin-bs1x1x8-may26-2"

class GenerateAnimation(bpy.types.Operator):
    """Generate Animation Script"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.generate_animation"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Generate hand animation"         # Display name in the interface.
    bl_options = {'REGISTER'} 
        
    conditioning_steps: bpy.props.IntProperty(name="# Conditioning Frames", default=30, min=1, max=100)
    new_frames: bpy.props.IntProperty(name="# Generated Frames", default=100, min=1)
    window_size: bpy.props.IntProperty(name="# Frames in Autoregressive Window", default=30, min=1, max=100)
    model_path: bpy.props.StringProperty(name="File path for model (dir)", default=model_path)
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):        # execute() is called when running the operator.
        global model_path
        wm = context.window_manager
        old_obj = context.object
        # new_obj = context.object.copy()
        model_path = self.model_path
        abs_model_path = bpy.path.abspath(model_path)
        
        # if not new_obj.name.find('generated') != -1:
        #     new_obj.name = old_obj.name + '.generated'
        
        animation_tracks = old_obj.animation_data.action.fcurves
        assert len(animation_tracks) == 54, f"Expected 54 total animation tracks on a hand BVH, found {len(animation_tracks)}"
        
        old_obj.animation_data.action = bpy.data.actions.new(name="GeneratedAction")
        

        conditioning_data = np.zeros([self.conditioning_steps, dof])
        
        i_track_n = 0
        for i_track, track in enumerate(animation_tracks):
            if i_track in USEFUL_COLUMNS:
                for i_frame, frame in enumerate(track.keyframe_points):
                    if i_frame >= self.conditioning_steps:
                        break
                    conditioning_data[i_frame, i_track_n] = frame.co.y
                i_track_n += 1
        
        generated_data = generate_data(wm, abs_model_path, conditioning_data, window_frames=self.window_size, new_frames=self.new_frames)
        
        n_frames = generated_data.shape[0]
        n_tracks = generated_data.shape[1]
        
        for i_track in range(n_tracks):
            track_name, track_type, track_index = COLUMN_ACCESS[i_track]
            data_path = f'pose.bones["{track_name}"].{track_type}'
            fc = old_obj.animation_data.action.fcurves.new(data_path=data_path, index=track_index)
            fc.keyframe_points.add(n_frames)
            for i_frame in range(n_frames):
                fc.keyframe_points[i_frame].co = (i_frame + 1.0, generated_data[i_frame, i_track])
        
        wm.progress_end()
                
        # context.collection.objects.link(new_obj)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def menu_func(self, context):
    self.layout.operator(GenerateAnimation.bl_idname)

def register():
    bpy.utils.register_class(VIEW3D_PT_generate_animation)
    bpy.utils.register_class(GenerateAnimation)

def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_generate_animation)
    bpy.utils.unregister_class(GenerateAnimation)

