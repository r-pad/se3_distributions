# -*- coding: utf-8 -*-
"""
Created on I think it was a friday
@author: bokorn
"""

from generic_pose.models.posecnn_mask import PoseCNNMask
from generic_pose.datasets.ycb_dataset import YCBDataset
import cv2
import os
import torchvision.transforms as transforms
import numpy as np

def main():

    #to_tensor = transforms.ToTensor()

    mask_net = PoseCNNMask('/home/bokorn/pretrained/pose_cnn/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt')      
    #mask_net.eval()
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/'
    dataset = YCBDataset(data_dir=benchmark_folder,
                         image_set='valid_split',
                         img_size=(224,224),
                         use_posecnn_masks = True,
                         obj=1)
    dataset.vgg_normalize = True
    dataset.background = 0.0
    dataset.loop_truth = None
    import IPython; IPython.embed()
    return 
    #prob = mask_net(to_tensor(img).unsqueeze(0))
    #mask = prob.data.permute(0,2,3,1).numpy()[0]
    for _ in range(100):
        j = np.random.randint(0,len(dataset))
        image_prefix = os.path.join(dataset.data_dir, 'data', dataset.data_filenames[0]) 
        #img = cv2.imread(image_prefix + '-color.png', cv2.IMREAD_UNCHANGED)
        img, _ = dataset.getImage(j)
        cv2.imwrite('/home/bokorn/results/test/ycb_masks/img_{:02}.png'.format(j), img) 
    #cv2.imwrite('/home/bokorn/results/test/ycb_masks/mask.png', mask*255) 
    #disp_mask_b = np.zeros(mask.shape[:2])
    #disp_mask_g = np.zeros(mask.shape[:2])
    #disp_mask_r = np.zeros(mask.shape[:2])
    #for j in range(mask.shape[2]):
    #    cv2.imwrite('/home/bokorn/results/test/ycb_masks/mask_{:02d}.png'.format(j), np.exp(mask[:,:,j])*255) 

    import IPython; IPython.embed()
if __name__=='__main__':
    main()

