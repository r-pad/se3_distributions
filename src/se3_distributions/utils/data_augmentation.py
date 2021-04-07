import numpy as np
import cv2
import PIL
import torchvision
from se3_distributions.utils.image_preprocessing import cropAndPad

import quat_math

def applyOcclusion(img, occlusion_area, color = 0):
    max_w, max_h = img.shape[:2]
    ratio = np.clip(np.random.rand()/np.random.rand(), 0.5, 2.0)
    area =  max_w * max_h * (np.random.rand()*(occlusion_area[1]-occlusion_area[0]) + occlusion_area[0])
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
    return 

def rotateAndScale(img, theta, scaleFactor = 1.0):
    (oldY,oldX) = img.shape[:2] 
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=np.rad2deg(theta), scale=scaleFactor)

    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    newX,newY = (abs(sin_theta*newY) + abs(cos_theta*newX),abs(sin_theta*newX) + abs(cos_theta*newY))

    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx 
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg

def rotateData(img, quat):
    theta = 2.0*np.pi*np.random.rand()
    quat_rot = quat_math.quaternion_multiply(quat_math.quaternion_about_axis(theta, [0,0,-1]), quat)
    img_rot = rotateAndScale(img, theta)
    return img_rot, quat_rot 

def augmentData(img, quat, 
                brightness_jitter = 0, contrast_jitter = 0, 
                saturation_jitter = 0, hue_jitter = 0,
                max_translation = None, max_scale = None,
                rotate_image = False, 
                max_num_occlusions = 0, max_occlusion_area = (0,0),
                transform_prob = 1.0, 
                max_occlusion_percent = 0.5, max_iter = 10):

    original_mask_area = float(np.sum(img[:,:,-1:]))
    num_iter = 0
    while(num_iter < max_iter):
        if(rotate_image): #and np.random.rand() < transform_prob):
            img, quat = rotateData(img, quat)

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
        if(mask_area / original_mask_area > 1-max_occlusion_percent):
            return cropAndPad(aug_img), quat 
        num_iter += 1
        #print('Failure {}: Augmentation yielded mask below threshold size ({:.2f} < {:.2f}) for original size of {:d}'.format(num_iter, mask_area / original_mask_area, max_occlusion_percent, int(original_mask_area)))
    print('Augmentation reached max iteration. Returning original image')
    return img, quat
