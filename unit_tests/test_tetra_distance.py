from model_renderer.pose_renderer import BpyRenderer
print('\n'*5)

from generic_pose.bbTrans.discretized4dSphere import S3Grid
from generic_pose.eval.exemplar_pose_estimator import topTetrahedron, refineTetrahedron, insideTetra, insideTetra1, insideTetra2
from generic_pose.utils.pose_processing import quatAngularDiffBatch 
from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from quat_math import random_quaternion, quatAngularDiff, projectedAverageQuaternion
from generic_pose.utils.image_preprocessing import unprocessImages


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shutil import copyfile
import numpy as np
import cv2
import os

def main():
    import time
    #np.random.seed(177)
    seed = np.random.randint(255)
    print(seed)
    #seed = 218
    np.random.seed(seed)
    level=0
    grid = S3Grid(level)
    grid.Simplify()
    
    num_samples = 100
    num_levels = 10
    level_dist = np.zeros(num_levels+1)
    inside_count = 0

    target_object = 15
    dataset = YCBDataset(data_dir='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset', 
                         image_set='train_split',
                         img_size=(224, 224),
                         obj=target_object)
    
    dataset.loop_truth = [1]
    
    renderer = BpyRenderer(transform_func = ycbRenderTransform)
    renderer.loadModel(dataset.getModelFilename(), emit = 0.5)
    renderPoses = renderer.renderPose
 
    for iter_num in range(num_samples):
        idx = np.random.randint(len(dataset))
        print(idx)
        tgt_quat = dataset.getQuat(idx)
        if(tgt_quat[3] < 0):
            tgt_quat *= -1
        
        #tgt_idx = np.random.randint(grid.GetTetras(2).shape[0])
        #tgt_tetra = grid.GetTetrahedron(tgt_idx, level=2)
        #tgt_quat = averageQuaternions(tgt_tetra.vertices) 

        def distFunc(quats):
            return -quatAngularDiffBatch(np.tile(tgt_quat, (np.shape(quats)[0],1)), quats)
            #return np.expand_dims(tgt_quat,0).dot(np.array(quats).T).flatten()
            #return -np.minimum(np.linalg.norm(tgt_quat - np.array(quats), axis=1),
            #                   np.linalg.norm(tgt_quat + np.array(quats), axis=1))
        def metricFunc(dists):
            return np.mean(dists)
            #return np.max(dists)
        
        dists = distFunc(grid.vertices)
        vertex_idx = np.argmax(dists)
        tetras = grid.GetTetras(level) 
        #t = time.time()
        tetra_idx = topTetrahedron(dists, tetras)
        #print(time.time()-t)
        tetra = grid.GetTetrahedron(tetra_idx)
        inside_count += insideTetra1(tetra, tgt_quat) or insideTetra1(tetra, -tgt_quat)
        #inside_count += insideTetra2(tetra, tgt_quat) or insideTetra2(tetra, -tgt_quat)
        ds = []
        #print('Pre: ', insideTetra2(tetra, tgt_quat) or insideTetra2(tetra, -tgt_quat))
        
        qs = []
        for j in range(num_levels+1):
            est_quat = refineTetrahedron(tgt_quat, tetra, distFunc, metricFunc, levels=j)
            d = quatAngularDiff(tgt_quat, est_quat)*180.0/np.pi
            level_dist[j] += d
            ds.append(d)
            qs.append(est_quat)
        print(ds)
        #print('insideTetra1') 
        #print(iter_num, tetra_idx, insideTetra1(tetra, tgt_quat) or insideTetra1(tetra, -tgt_quat), tgt_quat)
        #for j in range(tetras.shape[0]):           
        #    if(insideTetra1(grid.GetTetrahedron(j, level=level), tgt_quat)): 
        #        print(j) 
        #    if(insideTetra1(grid.GetTetrahedron(j, level=level), -tgt_quat)): 
        #        print(j)
        #print('insideTetra2') 
        #print(iter_num, tetra_idx, insideTetra2(tetra, tgt_quat) or insideTetra2(tetra, -tgt_quat), tgt_quat)
        #for j in range(tetras.shape[0]):
        #    if(insideTetra2(grid.GetTetrahedron(j, level=level), tgt_quat)): 
        #        print(j) 
        #    if(insideTetra2(grid.GetTetrahedron(j, level=level), -tgt_quat)): 
        #        print(j)
        #level_dist /= num_samples
        renderPoses(qs, camera_dist=0.33, image_filenames = ['/home/bokorn/results/ycb_finetune/tetra_imgs/{}.png'.format(k) for k in range(len(qs))])
        np.savetxt('/home/bokorn/results/ycb_finetune/tetra_imgs/error.txt', ds, fmt='%0.2f')
        np.savez('/home/bokorn/results/ycb_finetune/tetra_imgs/error.npz', errors=ds, quats = qs)
        plt.plot(np.arange(num_levels+1), ds)
        plt.ylim(bottom=0)
        plt.title('Tetrahedron Subdivision Accuracy')
        plt.xlabel('Tetrahedron Subdivision Level')
        plt.ylabel('Quaternion Error (deg)')
        plt.savefig('/home/bokorn/results/ycb_finetune/tetra_imgs/errors.png')
        plt.gcf().clear()
        #import IPython; IPython.embed()
        img = dataset.getImage(idx) 
        cv2.imwrite('/home/bokorn/results/ycb_finetune/tetra_imgs/target.png', unprocessImages(img.unsqueeze(0))[0])
        img_filename = os.path.join(dataset.data_dir, 'data', dataset.data_filenames[idx]+'-color.png')
        
        copyfile(img_filename, '/home/bokorn/results/ycb_finetune/tetra_imgs/orig.png') 
        import IPython; IPython.embed()

    print('\n'*5)

if __name__=='__main__':
    main()

