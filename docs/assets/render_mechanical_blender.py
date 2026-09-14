"""Build and render nfly's mechanical fly in Blender (no generated view sprites).

blender --background --python docs/assets/render_mechanical_blender.py -- --preview
blender --background --python docs/assets/render_mechanical_blender.py -- --frames 96
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import render_fly_blender as geometry

TAU = math.tau
OUT = Path('/tmp/nfly-mechanical-blender')
MAT = {}


def metal(name, color, roughness=0.27):
    return geometry.material(name, color, 0.88, roughness)


def luminous(name, color, strength):
    mat = geometry.material(name, color, 0.35, 0.22)
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Emission Color'].default_value = (*color, 1)
    shader.inputs['Emission Strength'].default_value = strength
    return mat


def cylinder(name, a, b, radius, material, sides=32):
    a, b = Vector(a), Vector(b)
    direction = b-a
    bpy.ops.mesh.primitive_cylinder_add(vertices=sides, radius=radius, depth=direction.length,
                                      location=(a+b)/2)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    obj.data.materials.append(material)
    obj.parent = geometry.ROOT
    bevel = obj.modifiers.new('Machined edge bevel', 'BEVEL')
    bevel.width = min(0.009, radius*0.13)
    bevel.segments = 3
    for face in obj.data.polygons:
        face.use_smooth = len(face.vertices) == 4
    return obj


def washer(name, pos, normal, radius, thickness, material):
    bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=10,
                                   location=pos, major_radius=radius, minor_radius=thickness)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = Vector(normal).to_track_quat('Z', 'Y').to_euler()
    obj.data.materials.append(material)
    obj.parent = geometry.ROOT
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


def screw(pos, normal, radius=0.018):
    pos, normal = Vector(pos), Vector(normal).normalized()
    cylinder('Recessed hex fastener', pos-normal*0.007, pos+normal*0.006, radius, MAT['silver'], 6)
    cylinder('Hex socket', pos+normal*0.006, pos+normal*0.007, radius*0.48, MAT['black'], 6)


def ellipsoid_point(center, radii, phi, theta):
    unit = Vector((math.sin(phi)*math.cos(theta), math.sin(phi)*math.sin(theta), math.cos(phi)))
    point = Vector(center) + Vector(tuple(r*u for r, u in zip(radii, unit)))
    normal = Vector(tuple(u/r for r, u in zip(radii, unit))).normalized()
    return point, normal


def armor_patch(center, radii, bounds, material):
    p0, p1, t0, t1 = bounds
    vertices, faces = [], []
    for i in range(7):
        for j in range(9):
            point, _ = ellipsoid_point(center, radii, p0+(p1-p0)*i/6, t0+(t1-t0)*j/8)
            vertices.append(point)
    for i in range(6):
        for j in range(8):
            a = i*9+j
            faces.append((a, a+9, a+10, a+1))
    obj = geometry.mesh_object('Individual titanium armor plate', vertices, faces, material)
    solidify = obj.modifiers.new('Plate thickness', 'SOLIDIFY')
    solidify.thickness = 0.018
    bevel = obj.modifiers.new('Reflective armor bevel', 'BEVEL')
    bevel.width = 0.008
    bevel.segments = 3
    for p, t in ((p0+0.065, t0+0.08), (p1-0.065, t1-0.08)):
        point, normal = ellipsoid_point(center, radii, p, t)
        screw(point+normal*0.01, normal, 0.014)


def armored_body(center, radii, rows=4, columns=8):
    geometry.sphere('Dark structural core', center, tuple(r*0.96 for r in radii), MAT['black'])
    for row in range(rows):
        for col in range(columns):
            phi0 = 0.10+(math.pi-0.2)*row/rows
            phi1 = 0.10+(math.pi-0.2)*(row+1)/rows
            theta = TAU*col/columns
            material = MAT['titanium'] if (row+col)%3 else MAT['steel']
            armor_patch(center, radii, (phi0+0.016, phi1-0.016, theta+0.025,
                                       theta+TAU/columns-0.025), material)


def thorax():
    center, radii = (0, 0.0, 0.38), (0.49, 0.63, 0.49)
    armored_body(center, radii)
    washer('Dorsal processor collar', (0, 0.02, 0.861), (0, 0, 1), 0.11, 0.016, MAT['silver'])
    cylinder('Dorsal photonic processor', (0, 0.02, 0.851), (0, 0.02, 0.87), 0.09, MAT['cyan'])
    for side in (-1, 1):
        # Parallel ventilation slits conform to the body surface.
        for i in range(9):
            theta = side*0.10 + i*0.042
            points = [ellipsoid_point(center, (0.499, 0.643, 0.499), p, theta)[0]
                      for p in (0.85, 0.98, 1.1)]
            geometry.curves('Thoracic heat-exchanger slot', [points], 0.009, MAT['black'])
        paths = []
        for k in range(3):
            theta = side*(0.8+k*0.09)
            paths.append([ellipsoid_point(center, (0.497, 0.639, 0.499), p, theta)[0]
                          for p in (0.3, 0.45, 0.65, 0.85)])
        geometry.curves('Embedded photonic bus', paths, 0.003, MAT['cyan'])


def abdomen_point(t, angle):
    radius = 0.49*(math.sin(math.pi*(0.12+0.875*t))**0.65)
    return Vector((radius*math.cos(angle), 0.31+1.48*t,
                   0.22+radius*0.76*math.sin(angle)-0.12*t))


def abdomen():
    geometry.sphere('Abdominal inner chassis', (0, 1.02, 0.18), (0.4, 0.73, 0.3), MAT['black'])
    for segment in range(7):
        low, high = segment/7+0.008, (segment+1)/7-0.008
        vertices, faces = [], []
        for i in range(9):
            for j in range(65):
                vertices.append(abdomen_point(low+(high-low)*i/8, TAU*j/64))
        for i in range(8):
            for j in range(64):
                a = i*65+j
                faces.append((a, a+1, a+66, a+65))
        obj = geometry.mesh_object('Overlapping abdominal armor band', vertices, faces, MAT['titanium'])
        thickness = obj.modifiers.new('Band thickness', 'SOLIDIFY')
        thickness.thickness = 0.016
        paths = [[abdomen_point(high, TAU*j/96) for j in range(97)]]
        geometry.curves('Polished band edge', paths, 0.012, MAT['silver'])
        for angle in (0.4, 1.1, 2.04, 2.74):
            pos = abdomen_point((low+high)/2, angle)
            screw(pos, (math.cos(angle), 0, math.sin(angle)), 0.016)
        for side in (-1, 1):
            for k in range(4):
                angle = (0 if side == 1 else math.pi)+0.05+k*0.09
                points = [abdomen_point(t, angle) for t in (low+0.02, high-0.02)]
                geometry.curves('Abdominal cooling grille', [points], 0.008, MAT['black'])
    washer('Rear service port', (0, 1.784, 0.1), (0, 1, 0), 0.042, 0.012, MAT['silver'])
    cylinder('Rear service indicator', (0, 1.78, 0.1), (0, 1.795, 0.1), 0.028, MAT['cyan'])


def eye(side):
    center, radii = Vector((side*0.345, -0.955, 0.44)), (0.28, 0.335, 0.345)
    geometry.sphere('Compound optical housing', center, radii, MAT['copper'])
    vertices, faces, indices = [], [], []
    for row in range(29):
        phi = 0.06+(math.pi-0.12)*row/28
        count = max(5, round(58*math.sin(phi)))
        for col in range(count):
            theta = TAU*(col+0.5*(row%2))/count
            p, normal = ellipsoid_point(center, radii, phi, theta)
            tangent = normal.cross(Vector((0, 0, 1))).normalized()
            bitangent = normal.cross(tangent).normalized()
            base = len(vertices)
            vertices.append(p+normal*0.009)
            for k in range(6):
                angle = TAU*k/6
                vertices.append(p+(tangent*math.cos(angle)+bitangent*math.sin(angle))*0.0144)
            for k in range(6):
                faces.append((base, base+1+k, base+1+(k+1)%6))
                indices.append((row*7+col*3)%4)
    obj = geometry.mesh_object('Thousands of amber hexagonal optical lenses', vertices, faces, MAT['lens0'])
    for i in range(1, 4):
        obj.data.materials.append(MAT[f'lens{i}'])
    for poly, index in zip(obj.data.polygons, indices):
        poly.material_index = index
        poly.use_smooth = False


def head():
    armored_body((0, -0.81, 0.4), (0.40, 0.34, 0.36), 3, 6)
    for side in (-1, 1):
        eye(side)
        a, b = (side*0.12, -1.18, 0.49), (side*0.19, -1.32, 0.5)
        cylinder('Antennal gimbal', a, b, 0.057, MAT['titanium'])
        washer('Antennal brass bearing', b, Vector(b)-Vector(a), 0.045, 0.012, MAT['copper'])
        points = [Vector(b), Vector((side*0.4, -1.47, 0.62)), Vector((side*0.64, -1.5, 0.67))]
        geometry.curves('Segmented sensor antenna', [points], 0.016, MAT['silver'])
        for k in range(1, 7):
            p = points[0].lerp(points[1], k/7)
            washer('Antenna insulation ring', p, points[1]-points[0], 0.017, 0.003, MAT['black'])
        geometry.sphere('Amber antenna sensor', points[-1], (0.026,)*3, MAT['amber'])
    cylinder('Central camera barrel', (0, -1.11, 0.19), (0, -1.24, 0.19), 0.105, MAT['black'])
    washer('Central camera bezel', (0, -1.25, 0.19), (0, -1, 0), 0.083, 0.015, MAT['silver'])
    cylinder('Central camera optic', (0, -1.247, 0.19), (0, -1.255, 0.19), 0.064, MAT['cyan'])
    for x in (-0.08, 0, 0.08):
        cylinder('Forehead status light', (x, -1.1, 0.69), (x, -1.115, 0.69), 0.018, MAT['cyan'])


def joint(pos, side, radius):
    pos = Vector(pos)
    axis = Vector((side, 0, 0))
    cylinder('Rotary actuator body', pos-axis*radius*0.7, pos+axis*radius*0.7, radius, MAT['black'])
    face = pos+axis*radius*0.76
    washer('Actuator brass bearing', face, axis, radius*0.75, radius*0.13, MAT['copper'])
    cylinder('Actuator end cap', face, face+axis*0.009, radius*0.55, MAT['steel'])
    screw(face+axis*0.012, axis, radius*0.28)


def leg_segment(a, b, radius):
    a, b = Vector(a), Vector(b)
    axis = (b-a).normalized()
    cylinder('Hexagonal titanium link', a, a.lerp(b, 0.65), radius, MAT['titanium'], 6)
    cylinder('Exposed chrome piston', a.lerp(b, 0.52), b, radius*0.52, MAT['silver'])
    for t in (0.12, 0.57, 0.64):
        washer('Piston collar', a.lerp(b, t), axis, radius, radius*0.15, MAT['copper'])
    offset = axis.cross(Vector((0, 0, 1))).normalized()*radius*0.95
    cylinder('Parallel hydraulic line', a+offset, b+offset, radius*0.14, MAT['copper'], 16)
    geometry.curves('Embedded link indicator', [[a.lerp(b, 0.19)+Vector((0, 0, radius)),
                                                a.lerp(b, 0.48)+Vector((0, 0, radius))]],
                    0.004, MAT['cyan'])


def legs(side):
    for y, knee_y, foot_y in ((-0.38, -0.74, -1.43), (0.03, 0.16, 0.22), (0.43, 1.04, 1.67)):
        points = [Vector((side*0.36, y, 0.23)), Vector((side*0.77, knee_y, -0.02)),
                  Vector((side*1.12, foot_y, -0.61)), Vector((side*1.35, foot_y-0.11, -0.72))]
        for i, (a, b) in enumerate(zip(points, points[1:])):
            joint(a, side, (0.105, 0.084, 0.05)[i])
            leg_segment(a, b, (0.067, 0.052, 0.027)[i])
        tip = points[-1]
        joint(tip, side, 0.039)
        for offset in (-0.037, 0.037):
            end = tip+Vector((side*0.1, offset-0.05, -0.015))
            cylinder('Split machined foot', tip, end, 0.019, MAT['steel'], 6)
            geometry.sphere('Rubber contact pad', end, (0.036, 0.03, 0.016), MAT['black'])


def wing_point(side, u, v):
    width = 0.40*math.sin(math.pi*u)**0.72
    return Vector((side*(0.35+1.13*u+width*v), 0.03+1.92*u-0.3*width*v, 0.74+0.14*u+0.025*v))


def wings(side):
    vertices, faces = [], []
    for i in range(65):
        for j in range(17):
            vertices.append(wing_point(side, i/64, j/8-1))
    for i in range(64):
        for j in range(16):
            a = i*17+j
            faces.append((a, a+1, a+18, a+17))
    geometry.mesh_object('Transparent photonic wing', vertices, faces, MAT['wing'])
    for v in (-1, 1):
        points = [wing_point(side, i/64, v) for i in range(65)]
        geometry.curves('Carbon wing perimeter', [points], 0.013, MAT['steel'])
        geometry.curves('Luminous wing edge', [[p+Vector((0, 0, 0.009)) for p in points]], 0.005, MAT['cyan'])
    paths = [[wing_point(side, i/64, v) for i in range(65)] for v in (-0.62, -0.2, 0.3, 0.68)]
    for u in (0.28, 0.51, 0.76):
        paths.append([wing_point(side, u+0.04*math.sin(j/16*math.pi), j/8-1) for j in range(17)])
    geometry.curves('Copper etched wing circuitry', paths, 0.005, MAT['copper'])
    fine = []
    for i in range(6, 61, 2):
        u = i/64
        fine.append([wing_point(side, u, v) for v in (-0.92, -0.7, -0.4, 0, 0.4, 0.7, 0.92)])
    geometry.curves('Fine photonic lattice', fine, 0.0012, MAT['cyan_dim'])
    for u in (0.28, 0.51, 0.76):
        for v in (-0.62, 0.3):
            geometry.sphere('Wing signal junction', wing_point(side, u, v), (0.012,)*3, MAT['amber'])
    joint((side*0.36, 0.04, 0.74), side, 0.09)
    cylinder('Wing hinge support', (side*0.32, 0.08, 0.57), (side*0.36, 0.04, 0.74), 0.045, MAT['silver'])
    cylinder('Stabilizer stalk', (side*0.42, 0.4, 0.28), (side*0.72, 0.59, 0.38), 0.025, MAT['silver'])
    geometry.sphere('Inertial sensor housing', (side*0.72, 0.59, 0.38), (0.075, 0.09, 0.07), MAT['titanium'])


def materials():
    MAT.update(titanium=metal('Black titanium armor', (0.035, 0.045, 0.055)),
               steel=metal('Gunmetal plate variation', (0.12, 0.15, 0.18)),
               silver=metal('Brushed silver edges', (0.48, 0.57, 0.64), 0.21),
               copper=metal('Warm brass bearings', (0.43, 0.22, 0.065), 0.23),
               black=geometry.material('Graphite recesses', (0.007, 0.01, 0.014), 0.35, 0.4),
               cyan=luminous('Cyan photonic emission', (0.008, 0.65, 1), 2.0),
               cyan_dim=luminous('Subtle wing lattice', (0.025, 0.2, 0.3), 0.6),
               amber=luminous('Amber indicators', (1, 0.22, 0.015), 2.0))
    for i, value in enumerate((0.3, 0.55, 0.8, 1.0)):
        lens = luminous(f'Amber compound lens {i}', (value, value*0.13, 0.003), 0.65)
        lens.node_tree.nodes.get('Principled BSDF').inputs['Metallic'].default_value = 0.7
        MAT[f'lens{i}'] = lens
    mat = geometry.material('Photonic glass membrane', (0.14, 0.48, 0.58), 0.45, 0.16)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    transparent, mix = nodes.new('ShaderNodeBsdfTransparent'), nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = 0.11
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(nodes.get('Principled BSDF').outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get('Material Output').inputs['Surface'])
    MAT['wing'] = mat


def scene_setup():
    scene = geometry.setup_scene()
    scene.cycles.samples = 40
    scene.render.use_persistent_data = True
    preferences = bpy.context.preferences.addons['cycles'].preferences
    if sys.platform == 'darwin':
        preferences.compute_device_type = 'METAL'
        preferences.get_devices()
        if any(device.type == 'METAL' for device in preferences.devices):
            for device in preferences.devices:
                device.use = device.type == 'METAL'
            scene.cycles.device = 'GPU'
    scene.camera.location = (4.0, -6.5, 3.1)
    scene.camera.rotation_euler = (Vector((0, 0.25, 0.18))-scene.camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera.data.ortho_scale = 6.0
    geometry.area('Front reflective softbox', (1, -4, 2), (0.85, 0.92, 1), 220, 2.5)
    return scene


def turntable_animation(scene, frames):
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.fps, scene.render.fps_base = 25, 2.0
    root = geometry.ROOT
    root.rotation_euler.z = 0
    root.keyframe_insert(data_path='rotation_euler', frame=1)
    root.rotation_euler.z = TAU
    root.keyframe_insert(data_path='rotation_euler', frame=frames+1)
    for layer in root.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = 'LINEAR'
    scene.frame_set(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--frames', type=int, default=96)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    geometry.ROOT = bpy.data.objects.new('Mechanical fly turntable', None)
    bpy.context.collection.objects.link(geometry.ROOT)
    materials()
    thorax()
    abdomen()
    head()
    for side in (-1, 1):
        legs(side)
        wings(side)
    scene = scene_setup()
    turntable_animation(scene, args.frames)
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / 'mechanical-fly.blend'), compress=True)
    for frame in range(1 if args.preview else args.frames):
        scene.frame_set(frame+1)
        scene.render.filepath = str(OUT / ('preview.png' if args.preview else f'frame-{frame:03d}.png'))
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
