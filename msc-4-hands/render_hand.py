import bpy

bpy.context.scene.objects['Cube'].select_set(True)
bpy.ops.object.delete()

bpy.ops.import_anim.bvh(filepath="./manipnet/Data/SimpleVisualizer/Assets/BVH/bottle1_body1/leftHand.bvh", use_fps_scale=True, update_scene_duration=True)

bpy.context.scene.objects['leftHand'].animation_data.action.fcurves[0].mute = True
bpy.context.scene.objects['leftHand'].animation_data.action.fcurves[1].mute = True
bpy.context.scene.objects['leftHand'].animation_data.action.fcurves[2].mute = True

# set wrist pose location to origin
bpy.context.scene.objects['leftHand'].pose.bones['WRIST'].location[0] = 0
bpy.context.scene.objects['leftHand'].pose.bones['WRIST'].location[1] = 0
bpy.context.scene.objects['leftHand'].pose.bones['WRIST'].location[2] = 0

bpy.context.scene.render.filepath = "/home/maxeonyx/msc/msc-cgt-hands/anims/"
bpy.context.scene.frame_end = 100
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

for area in bpy.context.screen.areas:
	if area.type == 'VIEW_3D':
		area.spaces[0].region_3d.view_perspective = 'CAMERA'

bpy.context.scene.objects['Camera'].data.lens = 94

bpy.ops.render.opengl(animation=True, view_context=True)

bpy.ops.wm.quit_blender()
