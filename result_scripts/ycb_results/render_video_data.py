import glob
import numpy as np
from generic_pose.eval.display_quaternions import plotQuatBall
from generic_pose.bbTrans.discretized4dSphere import S3Grid

data_folder = '/home/bokorn/results/ycb_finetune/video_results/'

training_files = glob.glob(data_folder + '*_train_results.npz')

train_filenames = sorted(glob.glob(data_folder + '*_train_results.npz'))
valid_filenames = sorted(glob.glob(data_folder + '*_valid_results.npz'))

train_data = {}

for fn in train_filenames:
    train_data['_'.join(fn.split('_')[:-2])] = np.load(fn)

valid_data = {}
for fn in valid_filenames:
    valid_data['_'.join(fn.split('_')[:-2])] = np.load(fn)

grid = S3Grid(2)
grid.Simplify()

import IPython; IPython.embed()

for k,v in train_data.items():
    dists = np.sum(v['agg_dists'], axis=0)
    plotQuatBall(grid.vertices, dists/np.max(dists), gt_quat = [0,0,0,1], 
                 img_prefix = k + '_train_distance_')

for k,v in valid_data.items():
    dists = np.sum(v['agg_dists'], axis=0)
    plotQuatBall(grid.vertices, dists/np.max(dists), gt_quat = [0,0,0,1], 
                 img_prefix = k + '_valid_distance_')

import IPython; IPython.embed()
