# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import sys
import math
import random
import numpy as np
import os, tempfile, glob, shutil

if __name__ != '__main__':
    import cv2

render_root_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, render_root_folder)
import transformations as tf

blank_blend_file_path = os.path.join(render_root_folder, 'blank.blend') 
#Should be a system variable
blender_executable_path = '/home/bokorn/src/blender/blender-2.79-linux-glibc219-x86_64/blender'
render_code = os.path.abspath(__file__)


def camera2quat(azimuth_deg, elevation_deg, tilt_deg):
    cx, cy, cz = objCentenedCameraPos(1, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, tilt_deg)
    q = quaternionProduct(q2, q1)
    return q

def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)
    
def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    #print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def objCentenedCameraPos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def renderView(model_file, pose_quats, camera_dist, filenames = None):
    temp_dirname = tempfile.mkdtemp()
    pose_quats = str([q.tolist() for q in pose_quats]).replace(',','').replace('[','').replace(']','')
    render_cmd = '{} {} --background --python {} -- --shape_file {} --pos_quats {} --camera_dist {} --image_folder {} > /dev/null 2>&1'.format(
            blender_executable_path, blank_blend_file_path, render_code, model_file, pose_quats, camera_dist, temp_dirname)
    
    try:
        os.system(render_cmd)
        image_filenames = sorted(glob.glob(temp_dirname+'/*.png'))
        
        rendered_imgs = []
        
        if(filenames is None):
            for render_filename in image_filenames:
                img = cv2.imread(render_filename, cv2.IMREAD_UNCHANGED)
                rendered_imgs.append(img)
        elif(len(image_filenames) == len(filenames)):
            for render_filename, save_filename  in zip(image_filenames, filenames):
                shutil.move(render_filename, save_filename)
        else:
            raise AssertionError('filenames must be None or same size as pose_quats, Expected {}, Got {}'.format(len(image_filenames), len(filenames)))

        shutil.rmtree(temp_dirname)
        
    except Exception as e:
        shutil.rmtree(temp_dirname)
        raise(e)
        
    return rendered_imgs


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    
    parser.add_argument('--shape_file', type=str, required=True)
    parser.add_argument('--pos_quats', nargs='+', type=float, required=True)
    parser.add_argument('--camera_dist', type=float, required=True)
    parser.add_argument('--image_folder', type=str, default='./test_images')
    
    parser.add_argument('--light_num_lower', type=int, default=1)
    parser.add_argument('--light_num_upper', type=int, default=6)
    parser.add_argument('--light_dist_lower', type=float, default=8)
    parser.add_argument('--light_dist_upper', type=float, default=20)
    parser.add_argument('--light_azimuth_lower', type=float, default=0)
    parser.add_argument('--light_azimuth_upper', type=float, default=360)
    parser.add_argument('--light_elevation_lower', type=float, default=-90)
    parser.add_argument('--light_elevation_upper', type=float, default=90)
    parser.add_argument('--light_energy_mean', type=float, default=2)
    parser.add_argument('--light_energy_std', type=float, default=2)
    parser.add_argument('--light_environment_energy_lower', type=float, default=0)
    parser.add_argument('--light_environment_energy_upper', type=float, default=1)

    args = parser.parse_args(sys.argv[6:])

    import bpy
    # Input parameters
    if args.image_folder is not None and not os.path.exists(args.image_folder):
        os.mkdir(args.image_folder)

    bpy.ops.import_scene.obj(filepath=args.shape_file) 
    
    bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    #bpy.context.scene.render.use_shadows = False
    #bpy.context.scene.render.use_raytrace = False
    
    bpy.data.objects['Lamp'].data.energy = 0
    
    #m.subsurface_scattering.use = True
    
    camObj = bpy.data.objects['Camera']
    # camObj.data.lens_unit = 'FOV'
    # camObj.data.angle = 0.2
        
    # set lights
    bpy.ops.object.select_all(action='TOGGLE')
    if 'Lamp' in list(bpy.data.objects.keys()):
        bpy.data.objects['Lamp'].select = True # remove default light
    bpy.ops.object.delete()

    assert len(args.pos_quats)%4==0, 'pos_quats must have multiple of 4 elements, Recieved {}'.format(len(args.pos_quats))

    num_quats = len(args.pos_quats)//4
    pose_quat_list = [np.array(args.pos_quats[(4*j):(4*j+4)]) for j in range(num_quats)]

    for pose_num, pos_quat in enumerate(pose_quat_list):
        camera_mat = np.eye(4)
        camera_mat[0,3] = args.camera_dist
        view_mat = tf.quaternion_matrix(tf.quaternion_inverse(pos_quat)).dot(camera_mat)
        cam_pos = view_mat[:3,3]              
        cam_quat = tf.quaternion_from_matrix(view_mat)
        # clear default lights
        bpy.ops.object.select_by_type(type='LAMP')
        bpy.ops.object.delete(use_global=False)
    
        # set environment lighting
        #bpy.context.space_data.context = 'WORLD'
        bpy.context.scene.world.light_settings.use_environment_light = True
        bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(args.light_environment_energy_lower, args.light_environment_energy_upper)
        bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
    
        # set point lights
        for i in range(random.randint(args.light_num_lower,args.light_num_upper)):
            light_azimuth_deg = np.random.uniform(args.light_azimuth_lower, args.light_azimuth_upper)
            light_elevation_deg  = np.random.uniform(args.light_elevation_lower, args.light_elevation_upper)
            light_dist = np.random.uniform(args.light_dist_lower, args.light_dist_upper)
            lx, ly, lz = objCentenedCameraPos(light_dist, light_azimuth_deg, light_elevation_deg)
            bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
            bpy.data.objects['Point'].data.energy = np.random.normal(args.light_energy_mean, args.light_energy_std)
    
        camera_mat = np.eye(4)
        
        camera_mat[0,3] = args.camera_dist
        view_mat = tf.quaternion_matrix(tf.quaternion_inverse(pos_quat)).dot(camera_mat)
        cam_pos = view_mat[:3,3]              
        ro_mat_pre = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        ro_mat_post = np.array([[0,0,1],[0,-1,0],[1,0,0]])
        cam_rot = ro_mat_pre.dot(view_mat[:3,:3].T.dot(ro_mat_post))
        cam_mat = np.eye(4)       
        cam_mat[:3,:3] = cam_rot
        cam_quat = tf.quaternion_from_matrix(cam_mat)
        
        camObj.location[0] = cam_pos[0]
        camObj.location[1] = cam_pos[1]
        camObj.location[2] = cam_pos[2]
        camObj.rotation_mode = 'QUATERNION'
        # Blender is [w,x,y,z]
        # Transform is [x,y,z,w]
        camObj.rotation_quaternion[0] = cam_quat[0]
        camObj.rotation_quaternion[1] = cam_quat[1]
        camObj.rotation_quaternion[2] = cam_quat[2]
        camObj.rotation_quaternion[3] = cam_quat[3]

        syn_image_file = '{}.png'.format(pose_num)
        bpy.data.scenes['Scene'].render.filepath = os.path.join(args.image_folder, syn_image_file)
        bpy.ops.render.render( write_still=True )

if __name__=='__main__':
    main()
