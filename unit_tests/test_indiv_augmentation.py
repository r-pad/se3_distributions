import os
import cv2
import numpy as np
from generic_pose.datasets.ycb_dataset import YCBDataset 
from generic_pose.utils.data_augmentation import applyOcclusion, augmentData
from generic_pose.utils.image_preprocessing import cropAndPad

from generic_pose.utils.image_preprocessing import unprocessImages

benchmark_folder = '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset'
test_dir = '/home/bokorn/results/test/aug/'
if(False):
    ycb_dataset = YCBDataset(data_dir=benchmark_folder,                                         
                             image_set='train_split',
                             img_size=(224, 224),                                    
                             obj=1)


    index = 0                                                                 
    image_prefix = os.path.join(ycb_dataset.data_dir, 'data', ycb_dataset.data_filenames[index])
    img = cv2.imread(image_prefix + '-color.png')    
    mask = 255*(cv2.imread(image_prefix + '-label.png')[:,:,:1] == 1).astype('uint8')
    img = np.concatenate([img, mask], axis=2)

    occ = mask.copy()
    occ = np.ones_like(mask)*255 
    for _ in range(10):
        applyOcclusion(occ, 0.1)

    cv2.imwrite(test_dir + 'occ.png', occ)

    cv2.imwrite(test_dir + 'img.png', img)
    img = cropAndPad(img)
    cv2.imwrite(test_dir + 'pad.png', img)
    cv2.imwrite(test_dir + 'none.png', augmentData(img, None)[0])
    cv2.imwrite(test_dir + 'brightness.png', augmentData(img, None, brightness_jitter=0.5)[0])
    cv2.imwrite(test_dir + 'contrast.png', augmentData(img, None, contrast_jitter=0.5)[0])
    cv2.imwrite(test_dir + 'saturation.png', augmentData(img, None, saturation_jitter=0.5)[0])
    cv2.imwrite(test_dir + 'hue.png', augmentData(img, None, hue_jitter=0.25)[0])
    cv2.imwrite(test_dir + 'translation.png', augmentData(img, None, max_translation = (0.2, 0.2))[0])
    cv2.imwrite(test_dir + 'crop.png', augmentData(img, None, max_crop = .2)[0])
    cv2.imwrite(test_dir + 'scale.png', augmentData(img, None, max_scale = (0.9, 1.1))[0])
    cv2.imwrite(test_dir + 'occlusions.png', augmentData(img, None, max_num_occlusions = 3, max_occlusion_area = (0.1, 0.3))[0])

    cv2.imwrite(test_dir + 'all.png', augmentData(img, None, brightness_jitter=0.5, 
                                                             contrast_jitter=0.5, 
                                                             saturation_jitter=0.5,
                                                             hue_jitter=0.25,
                                                             max_translation = (0.2, 0.2),
                                                             max_scale = (0.9, 1.1),
                                                             max_num_occlusions = 3, 
                                                             max_occlusion_area = 0.1)[0])


brightness_jitter = 0.5
contrast_jitter = 0.5
saturation_jitter = 0.5
hue_jitter = 0.25
max_translation = (0.2, 0.2)
max_scale = (0.8, 1.2)
rotate_image = True
max_num_occlusions = 3
max_occlusion_area = (0.1, 0.3)
augmentation_prob = 0.5

train_dataset = YCBDataset(data_dir=benchmark_folder, 
                           image_set='train_split',
                           img_size=(224, 224),
                           obj=1,
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

train_dataset.loop_truth = None
for j in range(10):
    idx = np.random.randint(len(train_dataset))
    train_dataset.augmentation_prob = augmentation_prob
    img, q = train_dataset.__getitem__(idx)[:2]
    cv2.imwrite(test_dir + '{}_aug.png'.format(j), 
            unprocessImages(img.unsqueeze(0))[0])
    train_dataset.augmentation_prob = 0
    img, q = train_dataset.__getitem__(idx)[:2]
    cv2.imwrite(test_dir + '{}_org.png'.format(j), 
            unprocessImages(img.unsqueeze(0))[0])

import IPython; IPython.embed()
