"""Detailed illustrative fly model; render with Blender in background mode.

Blender --background --python docs/assets/render_fly_blender.py -- --preview
Blender --background --python docs/assets/render_fly_blender.py -- --frames 96
Outputs PNG frames to /tmp/nfly-detailed. No biological data is used.
"""
import argparse
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

random.seed(18)
TAU = math.tau
OUT = Path('/tmp/nfly-detailed')


def material(name, color, metallic=0, roughness=0.4):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Metallic'].default_value = metallic
    shader.inputs['Roughness'].default_value = roughness
    return mat


def textured_shell(name, color):
    mat = material(name, color, 0.25, 0.34)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 110
    noise.inputs['Detail'].default_value = 3
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.22
    bump.inputs['Distance'].default_value = 0.018
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], nodes.get('Principled BSDF').inputs['Normal'])
    return mat


def mesh_object(name, vertices, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj.parent = ROOT
    return obj


def sphere(name, pos, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, location=pos)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    for face in obj.data.polygons:
        face.use_smooth = True
    obj.parent = ROOT
    return obj


def curves(name, paths, radius, mat):
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.resolution_u = 12
    data.bevel_depth = radius
    data.bevel_resolution = 2
    for points in paths:
        spline = data.splines.new('POLY')
        spline.points.add(len(points) - 1)
        for i, point in enumerate(points):
            spline.points[i].co = (*point[:3], 1)
            spline.points[i].radius = point[3] if len(point) > 3 else 1
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(mat)
    obj.parent = ROOT
    return obj


def bristles(center, scale, count, length):
    paths = []
    for _ in range(count):
        u, theta = random.uniform(-0.7, 1), random.uniform(0, TAU)
        normal = Vector((math.sqrt(1-u*u)*math.cos(theta), math.sqrt(1-u*u)*math.sin(theta), u))
        p = Vector(center) + Vector(tuple(s*n for s, n in zip(scale, normal)))
        tip = p + normal * random.uniform(length * 0.5, length)
        tip.y += length * 0.25
        paths.append([(*p, 1), (*(p.lerp(tip, 0.6)), 0.55), (*tip, 0.02)])
    curves('Tapered cuticular setae', paths, 0.004, HAIR)


def compound_eye(side):
    center = Vector((side*0.32, -0.88, 0.43))
    radii = (0.265, 0.315, 0.32)
    sphere('Ruby compound eye', center, radii, EYE)
    vertices, faces = [], []
    # Hexagonally staggered ommatidia on the outer half of each curved eye.
    for row in range(35):
        phi = 0.07 + (math.pi-0.14)*row/34
        count = max(5, round(70 * math.sin(phi)))
        for col in range(count):
            theta = TAU * (col+0.5*(row % 2))/count
            normal = Vector((math.sin(phi)*math.cos(theta), math.sin(phi)*math.sin(theta), math.cos(phi)))
            if normal.x * side < -0.12:
                continue
            p = center + Vector(tuple(r*n*1.006 for r, n in zip(radii, normal)))
            tangent = normal.cross(Vector((0, 0, 1))).normalized()
            bitangent = normal.cross(tangent).normalized()
            base = len(vertices)
            vertices.append(p + normal * 0.004)
            for k in range(6):
                a = TAU*k/6
                vertices.append(p + (tangent*math.cos(a) + bitangent*math.sin(a))*0.0114)
            for k in range(6):
                faces.append((base, base+1+k, base+1+(k+1)%6))
    mesh_object('Individual hexagonal eye facets', vertices, faces, FACET)


def abdomen():
    vertices, faces = [], []
    for i in range(81):
        t = i/80
        y = 0.26 + 1.5*t
        radius = 0.49 * (math.sin(math.pi*(0.12+0.87*t))**0.65)
        for j in range(96):
            a = TAU*j/96
            vertices.append((radius*math.cos(a), y, 0.23 + radius*0.73*math.sin(a)-0.13*t))
    for i in range(80):
        for j in range(96):
            faces.append((i*96+j, i*96+(j+1)%96, (i+1)*96+(j+1)%96, (i+1)*96+j))
    obj = mesh_object('Segmented amber abdomen', vertices, faces, AMBER)
    obj.data.materials.append(DARK)
    for face in obj.data.polygons:
        row = face.index//96
        if row % 13 >= 9 or row > 70:
            face.material_index = 1
    bristles((0, 0.95, 0.17), (0.43, 0.67, 0.31), 430, 0.05)


def legs(side):
    for y, knee_y, foot_y in ((-0.35, -0.8, -1.48), (0.02, 0.1, 0.25), (0.37, 1.05, 1.75)):
        points = [Vector((side*0.29, y, 0.15)), Vector((side*0.65, knee_y, -0.12)),
                  Vector((side*0.97, foot_y, -0.57)), Vector((side*1.21, foot_y+0.1, -0.7))]
        for i, (a, b) in enumerate(zip(points, points[1:])):
            curves('Articulated leg segment', [[(*a, 1), (*a.lerp(b, 0.2), 1.05), (*b, 0.57)]],
                   (0.052, 0.032, 0.017)[i], AMBER)
            sphere('Leg joint', a, (0.05-i*0.009,)*3, DARK)
            hairs = []
            for k in range(13):
                p = a.lerp(b, (k+1)/15)
                tip = p + Vector((side*0.055, 0.012, 0.045))
                hairs.append([(*p, 1), (*tip, 0.01)])
            curves('Leg sensory hairs', hairs, 0.003, HAIR)
        tip = points[-1]
        for k in range(5):
            end = tip + Vector((side*0.042, -0.055, -0.003))
            curves('Five tarsal segments', [[(*tip, 1), (*end, 0.65)]], 0.013-k*0.0014, AMBER)
            tip = end
        for offset in (-0.018, 0.018):
            curves('Paired pretarsal claws', [[tip, tip+Vector((side*0.05, offset, 0.025)),
                                            tip+Vector((side*0.065, offset, 0))]], 0.006, DARK)


def wing_point(side, u, v):
    width = 0.5*(math.sin(math.pi*u)**0.7)
    return (side*(0.29+1.05*u+width*v), 0.05+2.0*u-0.32*width*v, 0.66+0.1*u+0.025*v)


def wings(side):
    vertices, faces = [], []
    for i in range(65):
        for j in range(17):
            vertices.append(wing_point(side, i/64, j/8-1))
    for i in range(64):
        for j in range(16):
            a = i*17+j
            faces.append((a, a+1, a+18, a+17))
    mesh_object('Translucent wing membrane', vertices, faces, WING)
    paths = []
    for v in (-1, -0.56, -0.12, 0.37, 1):
        paths.append([wing_point(side, i/64, v) for i in range(65)])
    for u in (0.36, 0.65):
        paths.append([wing_point(side, u+0.035*math.sin(j/16*math.pi), j/8-1) for j in range(17)])
    curves('Longitudinal and cross veins', paths, 0.007, VEIN)
    micro = []
    for _ in range(1200):
        u, v = random.uniform(0.04, 0.96), random.uniform(-0.98, 0.98)
        p = Vector(wing_point(side, u, v))
        micro.append([(*p, 0.8), (*(p+Vector((0, 0.018, 0.007))), 0.01)])
    curves('Wing microtrichia', micro, 0.0015, VEIN)
    curves('Haltere stalk', [[(side*0.38, 0.39, 0.21), (side*0.7, 0.58, 0.34)]], 0.022, AMBER)
    sphere('Haltere knob', (side*0.7, 0.58, 0.34), (0.075, 0.09, 0.07), AMBER)


def head():
    sphere('Head capsule', (0, -0.81, 0.36), (0.4, 0.34, 0.35), SHELL)
    sphere('Facial plate', (0, -1.095, 0.24), (0.18, 0.065, 0.18), AMBER)
    sphere('Proboscis', (0, -1.11, 0.09), (0.1, 0.14, 0.1), DARK)
    for side in (-1, 1):
        compound_eye(side)
        sphere('Antennal pedicel', (side*0.1, -1.15, 0.42), (0.055, 0.07, 0.06), AMBER)
        sphere('Antennal funiculus', (side*0.15, -1.22, 0.4), (0.06, 0.1, 0.065), AMBER)
        start = Vector((side*0.17, -1.22, 0.45))
        end = start + Vector((side*0.32, -0.18, 0.3))
        paths = [[(*start, 1), (*end, 0.02)]]
        for k in range(1, 9):
            p = start.lerp(end, k/11)
            for branch in (-1, 1):
                tip = p + Vector((side*0.08, branch*0.09, 0.045))*(1-k/13)
                paths.append([(*p, 0.65), (*tip, 0.01)])
        curves('Feathered antennal arista', paths, 0.005, HAIR)
    for x, y in ((-0.06, -0.76), (0.06, -0.76), (0, -0.86)):
        sphere('Ocellus', (x, y, 0.69), (0.035, 0.04, 0.027), FACET)
    bristles((0, -0.81, 0.36), (0.4, 0.34, 0.35), 200, 0.07)


def area(name, location, color, energy, size):
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.object
    light.name = name
    light.data.color = color
    light.data.energy = energy
    light.data.shape = 'DISK'
    light.data.size = size
    light.rotation_euler = (Vector((0, 0.1, 0.2))-light.location).to_track_quat('-Z', 'Y').to_euler()


def setup_scene():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 680
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.world.color = (0.035, 0.035, 0.035)
    scene.view_settings.view_transform = 'AgX'
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.77))
    bpy.context.object.data.materials.append(material('Midnight backdrop', (0.006, 0.018, 0.028), 0.25, 0.32))
    bpy.ops.object.camera_add(location=(4.1, -6.2, 3.5))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0.3, 0.1))-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 5.0
    scene.camera = camera
    area('Softbox', (0, -3, 5), (0.8, 0.93, 1), 480, 4)
    area('Cyan rim', (2, 3, 2.5), (0.19, 0.8, 1), 650, 3)
    area('Warm fill', (-3, -1, 1.5), (1, 0.67, 0.4), 170, 2.5)
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--frames', type=int, default=96)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True, exist_ok=True)
    sphere('Thoracic scutum', (0, -0.02, 0.34), (0.43, 0.56, 0.42), SHELL)
    sphere('Scutellum', (0, 0.44, 0.41), (0.26, 0.24, 0.21), SHELL)
    bristles((0, -0.02, 0.34), (0.43, 0.56, 0.42), 850, 0.075)
    bristles((0, -0.02, 0.34), (0.43, 0.56, 0.42), 65, 0.18)
    abdomen()
    head()
    for side in (-1, 1):
        legs(side)
        wings(side)
    scene = setup_scene()
    for i in range(1 if args.preview else args.frames):
        ROOT.rotation_euler.z = TAU*i/args.frames
        scene.render.filepath = str(OUT / ('preview.png' if args.preview else f'frame-{i:03d}.png'))
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    ROOT = bpy.data.objects.new('Fly turntable', None)
    bpy.context.collection.objects.link(ROOT)
    SHELL = textured_shell('Bronze thoracic cuticle', (0.14, 0.105, 0.055))
    AMBER = textured_shell('Amber cuticle', (0.32, 0.17, 0.052))
    DARK = textured_shell('Dark abdominal bands', (0.035, 0.023, 0.014))
    HAIR = material('Fine golden bristles', (0.16, 0.12, 0.07), 0.1, 0.5)
    EYE = material('Deep red eye', (0.22, 0.015, 0.008), 0.15, 0.35)
    FACET = material('Copper red ommatidia', (0.43, 0.055, 0.018), 0.28, 0.29)
    VEIN = material('Wing veins', (0.29, 0.39, 0.34), 0.32, 0.3)
    WING = material('Thin wing membrane', (0.55, 0.75, 0.8), 0.05, 0.22)
    nodes, links = WING.node_tree.nodes, WING.node_tree.links
    shader = nodes.get('Principled BSDF')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = 0.16
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(shader.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get('Material Output').inputs['Surface'])
    main()
