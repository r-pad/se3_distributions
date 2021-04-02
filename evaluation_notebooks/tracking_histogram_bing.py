import sys 
import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import quat_math
import pickle

from PIL import Image
import scipy.io as scio
from functools import partial
from object_pose_utils.utils import to_np, to_var


from object_pose_utils.utils.interpolation import BinghamInterpolation, TetraInterpolation
from object_pose_utils.utils.multi_view_utils import applyTransformBatch

tetra_interp = TetraInterpolation(2)

def combineHistTetra(dists, trans, grid_vertices, confidence = None):
    num_dists = len(dists)
    if(confidence is None):
        confidence = np.ones(num_dists)
        
    grid_size = dists[0].shape
    joint_dist = np.ones(grid_size)
        
    for d, mat, w in zip(dists, trans, confidence):
        grid_vertices_inv = applyTransformBatch(grid_vertices, np.linalg.inv(mat))
        tetra_interp.setValues(d)
        v_interp = tetra_interp.smooth(grid_vertices_inv)
        
        joint_dist *= w*v_interp + (1-w)
        
    return joint_dist

grid_vertices = torch.load('/scratch/bokorn/results/posecnn_feat_all/grid/002_master_chef_can_vertices.pt')
bing_interp = BinghamInterpolation(grid_vertices.cuda(), sigma=torch.tensor(30.0).cuda())

def combineHistBingham(dists, trans, vertices, confidence = None):
    num_dists = len(dists)
    if(confidence is None):
        confidence = np.ones(num_dists)
        
    grid_size = dists[0].shape
    joint_dist = torch.ones(grid_size).cuda()
    for d, mat, w in zip(dists, trans, confidence):
        grid_vertices_inv = applyTransformBatch(vertices, np.linalg.inv(mat))
        
        bing_interp.setValues(d.flatten().cuda())
        v_interp = bing_interp(grid_vertices_inv.cuda())
        
        joint_dist *= w*v_interp + (1-w)
        
    return joint_dist

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--video_len', type=int, default = 3)
    parser.add_argument('--interval', type=int, default = 1)
    parser.add_argument('--mat_filename_format', type=str, default = 'results/{}/{}_fc6_results.mat')
    args = parser.parse_args()
    
    video_len = args.video_len
    interval = args.interval
    mat_filename_format = args.mat_filename_format
    save_key = mat_filename_format.split('}')[-1].split('.')[0]
    print(mat_filename_format)
    print(save_key)
    from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
    from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset as YCBVideoDataset
    from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes

    dataset_root = '/ssd0/datasets/ycb/YCB_Video_Dataset'

    with open('{0}/image_sets/classes.txt'.format(dataset_root)) as f:                                    
        classes = f.read().split()
    classes.insert(0, '__background__')

    object_list = list(range(1,22))
    mode = "test"

    output_format = [otypes.OBJECT_LABEL,
                     otypes.QUATERNION, 
                     ]

    ycb_dataset = YCBDataset(dataset_root, mode=mode, 
                             object_list = object_list, 
                             output_data = output_format, 
                             image_size = [640, 480], num_points=1000)

    dataset = YCBVideoDataset(ycb_dataset, 
                              interval = interval, 
                              video_len = video_len)

    grid_root = '/scratch/bokorn/results/posecnn_feat_all/'
    grid_vertices = torch.load(os.path.join(grid_root, 'grid',
                '{}_vertices.pt'.format(ycb_dataset.classes[1])))
    grid_size = grid_vertices.shape[0]

    from generic_pose.datasets.ycb_dataset import getYCBSymmeties
    from object_pose_utils.utils.pose_processing import symmetricAngularDistance, meanShift
    import pathlib


    import scipy.io as scio
    from tqdm import tqdm 

    grid_size = 3885

    err = {}
    lik = {}
    err_max_shift = {}

    with torch.no_grad():
        for obj_id in tqdm(object_list):
            sym_axis, sym_ang = getYCBSymmeties(obj_id)
            dataset.setObjectId(obj_id)
            
            err[obj_id]= {}
            lik[obj_id]= {}
            err_max_shift[obj_id]= {}
            for v_id in tqdm(dataset.getVideoIds()):
                dataset.setVideoId(v_id)
                err[obj_id][v_id] = []
                lik[obj_id][v_id] = []
                err_max_shift[obj_id][v_id] = []
                for j in tqdm(range(len(dataset))):
                    (data, trans) = dataset[j]
                    #obj, quat = data
                    if(len(data) < video_len):
                        err[obj_id][v_id].append(np.nan)
                        lik[obj_id][v_id].append(np.nan)
                        err_max_shift[obj_id][v_id].append(np.nan)
                        continue
                    quat = data[0][1]
                        
                    dists = []
                    
                    dists_found = True
                    for path in dataset.getPaths(j):
                        try:
                            dist_path = mat_filename_format.format(path, ycb_dataset.classes[obj_id])
                            dist_data = scio.loadmat(dist_path)
                            dists.append(torch.tensor(dist_data['dist_est'].flatten()).cuda())
                        except FileNotFoundError as e:
                            print(e)
                            err[obj_id][v_id].append(np.nan)
                            lik[obj_id][v_id].append(np.nan)
                            err_max_shift[obj_id][v_id].append(np.nan)
                            dists_found = False
                            continue
                    if(not dists_found):
                        continue

                    dist_joint = combineHistBingham(dists, trans, grid_vertices)
                    mode_quat = grid_vertices[torch.argmax(dist_joint)].unsqueeze(0)
                    v_shift = meanShift(mode_quat.cuda(), grid_vertices.cuda(), dist_joint.unsqueeze(1).float().cuda(),
                                        sigma=np.pi/9, max_iter = 100)
            
                    err[obj_id][v_id].append(symmetricAngularDistance(mode_quat, quat.unsqueeze(0),
                                          sym_axis, sym_ang).item()*180/np.pi)
                
                    err_max_shift[obj_id][v_id].append(symmetricAngularDistance(v_shift.cpu(), quat.unsqueeze(0),
                                                    sym_axis, sym_ang).item()*180/np.pi)

                    
                    savepath = 'results/{}'.format(dataset.getPaths(j)[0])
                    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)

                    
                    bing_interp.setValues(dist_joint.flatten())
                    lik[obj_id][v_id].append(bing_interp(quat.unsqueeze(0).cuda()).item())
                
                    if(False):
                        scio.savemat('{}/{}_{}_{}x{}_results.mat'.format(savepath, 
                                                               ycb_dataset.classes[obj_id], 
                                                               feature_key,
                                                               video_len, interval),
                                {'index':j,
                                 'quat':to_np(quat),
                                 'mode_quat':to_np(mode_quat),
                                 'dist_est':to_np(dist_joint),
                                 'lik':lik[obj_id][v_id][-1],
                                 'err':err[obj_id][v_id][-1],
                                 'err_max_shift':err_max_shift[obj_id][v_id][-1],
                                 })
    np.savez('results/hist_filter_{}_bint_{}x{}.npz'.format(save_key, video_len, interval), 
             lik=lik,
             err=err,
             err_max_shift=err_max_shift)

if __name__=='__main__':
    import socket
    import seuss_cluster_alerts as sca
    hostname = socket.gethostname()
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    if(gpu_id is not None):
        hostname += ' GPU {}'.format(gpu_id)

    try:
        main()
        sca.sendAlert('bokorn@andrew.cmu.edu', 
                       message_subject='Job Completed on {}'.format(hostname))

    except:
        e = sys.exc_info()
        sca.sendAlert('bokorn@andrew.cmu.edu', 
                message_subject='Job Failed on {}'.format(hostname),
                message_text=str(e))
        raise
