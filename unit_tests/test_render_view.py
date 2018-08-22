"""
@author edwardahn
"""

from generic_pose.utils.syscall_renderer import renderView
import numpy as np
import time

def testRenderView():
    camera_dist = 5.7869
    pose = np.ones(4)
    pose[0] = 0.7622
    pose[1] = 0.1931
    pose[2] = -0.2633
    pose[3] = 0.5589
    pose_quats = [pose]*2
    model_file = '/scratch/bokorn/data/models/035_power_drill/google_64k/textured.obj'

    t = time.time()
    img2 = renderView(model_file, pose_quats, camera_dist, debug_mode=False)
    print('Render time 2: ', time.time() - t)
    
    pose_quats = [pose]*20
    t = time.time()
    img20 = renderView(model_file, pose_quats, camera_dist, debug_mode=False)
    print('Render time 20: ', time.time() - t)

    pose_quats = [pose]*200
    t = time.time()
    img200 = renderView(model_file, pose_quats, camera_dist, debug_mode=False)
    print('Render time 20: ', time.time() - t)
    import IPython; IPython.embed()

if __name__ == '__main__':
    testRenderView()
