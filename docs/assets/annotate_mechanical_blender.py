"""Add editable, camera-facing software anatomy labels to the mechanical fly.

blender --background docs/assets/mechanical-fly.blend \
  --python docs/assets/annotate_mechanical_blender.py -- --preview
Use --frames 72 for the animated scene and PNG sequence.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

OUT = Path('/tmp/nfly-annotated')
WIDTH, HEIGHT = 1440, 900
SCALE = 9.8
COLORS = {}
CAMERA = None


def emission(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    shader = nodes.new('ShaderNodeEmission')
    shader.inputs['Color'].default_value = (*color, 1)
    shader.inputs['Strength'].default_value = 1.5
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(shader.outputs[0], output.inputs['Surface'])
    return material


def screen_point(x, y, depth=-3.8):
    return Vector(((x/WIDTH-0.5)*SCALE, (0.5-y/HEIGHT)*SCALE*HEIGHT/WIDTH, depth))


def mesh(name, points, faces, color, depth=-3.8):
    data = bpy.data.meshes.new(name)
    data.from_pydata([screen_point(x, y, depth) for x, y in points], [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.parent = CAMERA
    obj.data.materials.append(COLORS[color])
    obj.visible_shadow = False
    return obj


def rectangle(name, x, y, width, height, color, depth=-3.85):
    return mesh(name, [(x,y), (x+width,y), (x+width,y+height), (x,y+height)],
                [(0,1,2,3)], color, depth)


def line(name, points, color='dim', width=1.0):
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.bevel_depth = width*SCALE/WIDTH/2
    data.bevel_resolution = 2
    spline = data.splines.new('POLY')
    spline.points.add(len(points)-1)
    for point, (x,y) in zip(spline.points, points):
        point.co = (*screen_point(x,y),1)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.parent = CAMERA
    obj.data.materials.append(COLORS[color])
    obj.visible_shadow = False
    return obj


def text(name, body, x, y, size=22, color='text', bold=False):
    data = bpy.data.curves.new(name, 'FONT')
    data.body = body
    data.size = size*SCALE/WIDTH
    data.space_line = 1.2
    font = Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf')
    if font.is_file():
        data.font = bpy.data.fonts.load(str(font), check_existing=True)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.parent = CAMERA
    obj.location = screen_point(x,y)
    obj.data.materials.append(COLORS[color])
    obj.visible_shadow = False
    return obj


def panel(number, title, class_name, x, y, rows, accent):
    line(title+' accent', [(x,y), (x+320,y)], accent, 2)
    text(title+' number', number, x, y+34, 20, accent, True)
    text(title+' heading', title, x, y+75, 29, 'text', True)
    text(title+' class', class_name, x, y+108, 21, accent)
    for i, row in enumerate(rows):
        text(title+f' detail {i}', row, x, y+151+i*28, 20, 'muted')


def retina_diagram():
    x, y = 50, 565
    text('Retina diagram label', 'ILLUSTRATIVE IMAGE SAMPLING', x, y-22, 16, 'dim', True)
    rectangle('Example image', x, y, 115, 110, 'panel')
    rectangle('Pong paddle left', x+12, y+24, 5, 33, 'text', -3.81)
    rectangle('Pong paddle right', x+98, y+53, 5, 33, 'text', -3.81)
    rectangle('Pong ball', x+70, y+40, 7, 7, 'amber', -3.81)
    line('Frame border', [(x,y),(x+115,y),(x+115,y+110),(x,y+110),(x,y)], 'dim')
    line('Sampling arrow', [(x+131,y+54),(x+158,y+54),(x+152,y+48),
                            (x+158,y+54),(x+152,y+60)], 'cyan', 1.5)
    for row in range(6):
        for col in range(7):
            cx, cy = x+182+col*19+(row%2)*9.5, y+8+row*17
            points = [(cx+10*math.cos(math.tau*k/6+math.pi/6),
                       cy+10*math.sin(math.tau*k/6+math.pi/6)) for k in range(6)]
            highlight = (col == 1 and row in (1,2)) or (col == 5 and row in (3,4))
            color = 'amber' if (col,row)==(4,2) else ('cyan' if highlight else 'panel')
            mesh('Hexagonal sample', points, [tuple(range(6))], color, -3.84)
            line('Hexagonal boundary', points+[points[0]], 'dim', 0.7)
    text('Frame label', 'frame', x+32,y+138,18,'muted')
    text('Hex label', 'hex samples', x+194,y+138,18,'muted')
    text('Vector encoder note', 'Vector inputs use a linear encoder.', x,y+190,18,'muted')


def recurrent_diagram():
    x,y=1080,452
    for i,label in enumerate(('h[t-1]','h[t]','h[t+1]')):
        rectangle('Recurrent state box',x+i*109,y,88,43,'panel')
        text('Recurrent state label',label,x+12+i*109,y+29,20,'cyan')
        if i<2:
            line('State transition',[(x+91+i*109,y+22),(x+106+i*109,y+22)],'cyan',1.5)
    line('Recurrent feedback',[(x+148,y),(x+148,y-16),(x+178,y-16),(x+178,y),
                               (x+173,y-6)],'amber',1.5)


def labels():
    text('Title','nfly / anatomy of a policy',50,63,37,'text',True)
    text('Subtitle','Observation -> sensory encoding -> recurrent state -> action',50,104,23,'muted')
    line('Header rule',[(50,131),(1390,131)],'dim',1)
    panel('01 / EYES','Visual input','RetinaEncoder',50,208,
          ('Image -> grayscale [0, 1]','Hex-coordinate sampling','Per-eye grid in [-1, 1]',
           'Bilinear samples -> [-1, 1]'),'cyan')
    retina_diagram()
    panel('02 / BODY','Sparse recurrent network','ConnectomeRNN (rate model)',1080,174,
          ('Fixed wiring + synapse signs','Trainable edge gains','Trainable leak + bias',
           'One state value per neuron'),'cyan')
    recurrent_diagram()
    panel('03 / FEET','Action output','ActionDecoder',1080,546,
          ('Descending + motor readout','Discrete -> Categorical','Box -> Gaussian',
           'action -> env.step(action)'),'amber')
    text('Interface label','OPENAI GYM / GYMNASIUM',470,810,19,'cyan',True)
    text('Loop labels','observation_space                         action_space',470,841,18,'muted')
    text('Concept caption','Conceptual mechanical anatomy; labels describe software components.',50,879,17,'dim')


def linear_keys(data):
    if data.animation_data and data.animation_data.action:
        for layer in data.animation_data.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for key in curve.keyframe_points:
                            key.interpolation='LINEAR'


def anchors(scene, frames):
    root=bpy.data.objects['Mechanical fly turntable']
    specs=[('Eye input leader',Vector((0.345,-0.955,0.44)),[(378,350),(423,350)],'cyan'),
           ('Network body leader',Vector((0,0,0.86)),[(1063,320),(1025,320)],'cyan'),
           ('Foot action leader',Vector((1.35,-1.54,-0.72)),[(1063,665),(1025,665)],'amber')]
    for name,anchor,elbow,color in specs:
        obj=line(name,elbow+[(720,450)],color,1.3)
        point=obj.data.splines[0].points[-1]
        bpy.ops.mesh.primitive_torus_add(major_segments=32,minor_segments=8,
                                        major_radius=7*SCALE/WIDTH,minor_radius=SCALE/WIDTH)
        marker=bpy.context.object
        marker.name=name+' target ring'
        marker.parent=CAMERA
        marker.data.materials.append(COLORS[color])
        marker.visible_shadow=False
        for frame in range(1,frames+2):
            scene.frame_set(frame)
            uv=world_to_camera_view(scene,CAMERA,root.matrix_world@anchor)
            point.co=(*screen_point(uv.x*WIDTH,(1-uv.y)*HEIGHT),1)
            point.keyframe_insert(data_path='co',frame=frame)
            marker.location=screen_point(uv.x*WIDTH,(1-uv.y)*HEIGHT,-3.78)
            marker.keyframe_insert(data_path='location',frame=frame)
        linear_keys(obj.data)
        linear_keys(marker)


def setup(frames):
    global CAMERA
    scene=bpy.context.scene
    CAMERA=scene.camera
    CAMERA.data.ortho_scale=SCALE
    scene.render.resolution_x,scene.render.resolution_y=WIDTH,HEIGHT
    scene.render.resolution_percentage=100
    scene.frame_start,scene.frame_end=1,frames
    scene.render.fps,scene.render.fps_base=25,2
    scene.cycles.samples=24
    scene.render.use_persistent_data=True
    # A flat background keeps the labels clear and the animated GIF small.
    for obj in list(bpy.data.objects):
        if obj.type=='MESH' and any(mat and mat.name.startswith('Midnight backdrop') for mat in obj.data.materials):
            bpy.data.objects.remove(obj,do_unlink=True)
    scene.world.use_nodes=True
    background=scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value=(0.005,0.012,0.023,1)
    background.inputs['Strength'].default_value=0.3
    root=bpy.data.objects['Mechanical fly turntable']
    for layer in root.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    curve.keyframe_points[-1].co.x=frames+1
                    curve.update()
    linear_keys(root)
    palette={'text':(0.7,0.84,0.9),'muted':(0.34,0.47,0.56),'dim':(0.12,0.25,0.33),
             'cyan':(0.01,0.65,0.9),'amber':(1,0.30,0.04),'panel':(0.006,0.022,0.035)}
    COLORS.update({name:emission('HUD '+name,color) for name,color in palette.items()})
    labels()
    anchors(scene,frames)
    scene.frame_set(1)
    preferences=bpy.context.preferences.addons['cycles'].preferences
    if sys.platform=='darwin':
        preferences.compute_device_type='METAL'
        preferences.get_devices()
        for device in preferences.devices:
            device.use=device.type=='METAL'
        scene.cycles.device='GPU'
    bpy.ops.file.pack_all()
    return scene


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--frames',type=int,default=72)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True,exist_ok=True)
    scene=setup(args.frames)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'mechanical-fly-annotated.blend'),compress=True)
    for frame in range(1 if args.preview else args.frames):
        scene.frame_set(frame+1)
        scene.render.filepath=str(OUT/('preview.png' if args.preview else f'frame-{frame:03d}.png'))
        bpy.ops.render.render(write_still=True)


if __name__=='__main__':
    main()
