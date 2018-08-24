import time
import numpy as np
import model_renderer.pose_renderer
import torch
from torch.utils.data import DataLoader

from generic_pose.eval.hyper_distance import ExemplarDistPoseEstimator
from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.datasets.benchmark_dataset import LinemodDataset
from quat_math import (quatAngularDiff, 
                       quaternion_multiply, 
                       quaternion_inverse)
from generic_pose.utils import to_np


dist_net = gen_pose_net('alexnet', 'sigmoid', output_dim = 1, pretrained = False)

weight_file = '/home/bokorn/results/shapenet/distance/shapenet_exp_fo20_th25/2018-08-03_02-29-12/weights/checkpoint_86000.pth'
dist_net.load_state_dict(torch.load(weight_file))

model_folder = '/scratch/bokorn/data/benchmarks/linemod/iron/'
base_level = 2

def convertQuat(q):
    q_flip = quaternion_inverse(q.copy())
    #q_flip = quaternion_multiply(trans_quat, q_flip)
    q_flip[2] *= -1
    return quaternion_multiply(delta_quat, q_flip)

def trueDiff(q, ref_quats):
    q_adj = convertQuat(q)
    true_diff = []
    for v in ref_quats:
        true_diff.append(quatAngularDiff(q_adj, v))
    return np.array(true_diff)


pose_estimator = ExemplarDistPoseEstimator(model_filename = model_folder + 'mesh.ply',
                                           dist_network = dist_net,
                                           use_bpy_renderer = True,
                                           base_level = base_level)

data_loader = DataLoader(LinemodDataset(data_folders=model_folder,
                                        use_mask = True,
                                        img_size = (224, 224),
                                        max_orientation_offset = None,
                                        max_orientation_iters = None,
                                        model_filenames=None,
                                        background_filenames = None),
                         num_workers=0, 
                         batch_size=1, 
                         shuffle=True)

data_loader.dataset.loop_truth = [1]
img, _, quat, _, _ = next(iter(data_loader))
img = img[0]
quat = to_np(quat[0])
import IPython; IPython.embed()
