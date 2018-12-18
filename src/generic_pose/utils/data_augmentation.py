import numpy as np
import cv2
import PIL
import torchvision
from generic_pose.utils.image_preprocessing import cropAndPad

def applyOcclusion(img, max_occlusion_area, color = 0):
    max_w, max_h = img.shape[:2]
    max_area = max_occlusion_area * max_w * max_h
    ratio = np.random.rand()/np.random.rand();
    area = np.random.rand()*max_area
    h = np.sqrt(area*ratio)
    w = area/h

    xc = np.random.randint(max_w)
    yc = np.random.randint(max_h)
    angle = np.random.rand()*360

    if(np.random.randint(2)):
        box = np.int0(cv2.boxPoints(((xc, yc), (w, h), angle)))
        cv2.drawContours(img,[box],0,color, cv2.FILLED)
    else:
        sweep_angle = np.random.rand()*360;
        start_angle = np.random.rand()*360;
        end_angle = sweep_angle - start_angle;
        cv2.ellipse(img, (yc, xc), (int(h/2), int(w/2)), 
                    angle, start_angle, end_angle, 
                    color, cv2.FILLED)

def augmentData(img, quat, 
                brightness_jitter = 0, contrast_jitter = 0, 
                saturation_jitter = 0, hue_jitter = 0,
                max_translation = None, max_scale = None,
                rotate_image = False, 
                max_num_occlusions = 0, max_occlusion_area = 0,
                transform_prob = 1.0, 
                max_occlusion_percent = 0.5, max_iter = 10):

    original_mask_area = float(np.sum(img[:,:,-1:]))
    num_iter = 0
    while(num_iter < max_iter):
        mask = img[:,:,-1:].copy()
        if(max_num_occlusions):
            num_occlusions = np.random.randint(max_num_occlusions)
            for _ in range(num_occlusions):
                applyOcclusion(mask, max_occlusion_area) 

        toPil = torchvision.transforms.ToPILImage()
        img_pil = toPil(img[:,:,:3])

        if(brightness_jitter or contrast_jitter or saturation_jitter or hue_jitter \
                and np.random.rand() < transform_prob):
            color_jitter = torchvision.transforms.ColorJitter(brightness=brightness_jitter, 
                                                              contrast=contrast_jitter,
                                                              saturation=saturation_jitter, 
                                                              hue=hue_jitter)
            img_pil = color_jitter(img_pil)

        img_pil.putalpha(toPil(mask)) 
        if(max_translation or max_scale and np.random.rand() < transform_prob):
            random_affine = torchvision.transforms.RandomAffine(0, translate = max_translation,
                                                                scale = max_scale, fillcolor = 0)
            img_pil = random_affine(img_pil)
        
        #if(max_crop):
        #    crop_w = int(img.shape[0] * (1.0-np.random.rand()*max_crop))
        #    crop_h = int(img.shape[1] * (1.0-np.random.rand()*max_crop))
        #    transforms.append(torchvision.transforms.RandomCrop((crop_w, crop_h)))
        
        #trans = torchvision.transforms.RandomApply(transforms, transform_prob)
        aug_img = np.asarray(img_pil)
        mask_area = np.sum(aug_img[:,:,-1])
        if(mask_area / original_mask_area > max_occlusion_percent):
            return cropAndPad(aug_img), quat 
        num_iter += 1
        print('Failure {}: Augmentation yielded mask below threshold size ({:.2f} < {:.2f}) for original size of {:d}'.format(num_iter, mask_area / original_mask_area, max_occlusion_percent, int(original_mask_area)))
    print('Augmentation reached max iteration. Returning original image')
    return img, quat
