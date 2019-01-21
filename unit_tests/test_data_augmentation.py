import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from generic_pose.datasets.ycb_dataset import YCBDataset 
from generic_pose.utils.data_augmentation import applyOcclusion, augmentData
from generic_pose.utils.image_preprocessing import cropAndPad
from generic_pose.datasets.tensor_dataset import TensorDataset

from generic_pose.utils.image_preprocessing import unprocessImages
from quat_math import random_quaternion

benchmark_folder = '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset'
test_dir = '/home/bokorn/results/test/aug/'
renders_folder = '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/base_renders/002_master_chef_can/random_renders/' 
target_object = 1 
num_workers = 0
batch_size = 1

base_level = 2
augmentation_prob = 0.5 
brightness_jitter = 1.0 
contrast_jitter = 1.0 
saturation_jitter = 1.0 
hue_jitter = 0.25 
max_translation = (0.5, 0.5)
max_scale = (0.8, 1.2) 
max_num_occlusions = 3 
max_occlusion_area = (0.1, 0.3)
rotate_image = True
render_offset = random_quaternion()
 
ycb_dataset = YCBDataset(data_dir=benchmark_folder, 
                                        image_set='train_split',
                                        img_size=(224, 224),                                    
                                        obj=target_object,
                                        use_syn_data=True,
                                        brightness_jitter = brightness_jitter,
                                        contrast_jitter = contrast_jitter,
                                        saturation_jitter = saturation_jitter,
                                        hue_jitter = hue_jitter,
                                        max_translation = max_translation,
                                        max_scale = max_scale,
                                        rotate_image = rotate_image,
                                        max_num_occlusions = max_num_occlusions,
                                        max_occlusion_area = max_occlusion_area,
                                        augmentation_prob = augmentation_prob)

ycb_dataset.loop_truth = None
ycb_dataset.append_rendered = False

rendered_dataset = TensorDataset(renders_folder,
                                              ycb_dataset.getModelFilename(),
                                        img_size=(224, 224),                                    
                                              offset_quat = render_offset,
                                              base_level = base_level,
                                        brightness_jitter = brightness_jitter,
                                        contrast_jitter = contrast_jitter,
                                        saturation_jitter = saturation_jitter,
                                        hue_jitter = hue_jitter,
                                        max_translation = max_translation,
                                        max_scale = max_scale,
                                        rotate_image = rotate_image,
                                        max_num_occlusions = max_num_occlusions,
                                        max_occlusion_area = max_occlusion_area,
                                        augmentation_prob = augmentation_prob)

rendered_dataset.loop_truth = None
rendered_dataset.append_rendered = False
render_proportion = 1.0
ycb_multi = 1
render_multi = int(max(1,
np.ceil(render_proportion * len(ycb_dataset)/len(rendered_dataset))))
print('YCB Size: {}'.format(len(ycb_dataset)))
print('YCB Mulit: {}'.format(ycb_multi))
print('Render Size: {}'.format(len(rendered_dataset)))
print('Render Mulit: {}'.format(render_multi))
train_dataset = ConcatDataset((ycb_dataset,)*ycb_multi + (rendered_dataset,)*render_multi)
#train_loader = DataLoader(train_dataset,
#                                     num_workers=num_workers-1, 
#                                     batch_size=batch_size, 
#                                     shuffle=True)

for j in range(100):
    idx = np.random.randint(len(train_dataset))
    ycb_dataset.augmentation_prob = augmentation_prob
    rendered_dataset.augmentation_prob = augmentation_prob
    img_aug, q = train_dataset.__getitem__(idx)[:2]
    cv2.imwrite(test_dir + '{}_aug.png'.format(j), 
            unprocessImages(img_aug.unsqueeze(0))[0])
    ycb_dataset.augmentation_prob = 0.0
    rendered_dataset.augmentation_prob = 0.0
    img_org, q = train_dataset.__getitem__(idx)[:2]
    cv2.imwrite(test_dir + '{}_org.png'.format(j), 
            unprocessImages(img_org.unsqueeze(0))[0])

import IPython; IPython.embed()
