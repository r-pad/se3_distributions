# -*- coding: utf-8 -*-
"""
@author: bokorn
"""

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
from generic_pose.utils import to_np

def cropBBox(img, bbox, boarder_width = 10):
    rows, cols = img.shape[:2]
    x,y,w,h = bbox
    y0 = min(max(y - boarder_width, 0), rows)
    x0 = min(max(x - boarder_width, 0), cols)
    y1 = min(max(y + h + boarder_width, 0), rows)
    x1 = min(max(x + w + boarder_width, 0), cols)
    img_crop = img[y0:y1,x0:x1]

    return img_crop, (x0, y0)

def seg2Mask(segmentation, img_size):
    mask = np.zeros(img_size[:2] + (1,), dtype=np.uint8)
    polys = []
    for seg in segmentation:
        polys.append(np.reshape(seg,(-1,2)))
    cv2.fillPoly(mask, polys, (1))
    return mask

norm_mean = np.array([0.485, 0.456, 0.406])
#norm_std = np.array([0.229, 0.224, 0.225])

#normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
vgg_mean = torch.tensor([102.9801, 115.9465, 122.7717])
def vggToTensor(img, mean = vgg_mean):
    return (torch.tensor(img).float() - mean).permute([2,0,1])

def cropAndPad(img, padding_size = 0.1):
    where = np.array(np.where(img[:,:,3]))
    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)

    sub_img = img[x1:(x2+1), y1:(y2+1)]
    pad_size = int(max(sub_img.shape[:2])*padding_size)
    pad_img = cv2.copyMakeBorder(sub_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT,value=0)
    return pad_img

def unprocessImages(imgs,
                    norm_mean = np.array([0.485, 0.456, 0.406]),
                    norm_std = np.array([0.229, 0.224, 0.225])):
    imgs = np.transpose(to_np(imgs), (0,2,3,1))
    imgs = np.minimum(np.maximum(imgs*norm_std + norm_mean, 0.0), 1.0)*255
    return imgs

def preprocessImages(imgs, img_size,
                     normalize_tensors = False,
                     background = None,
                     background_filenames = None,
                     crop_percent = None,
                     remove_mask = True,
                     vgg_normalize = False):
    p_imgs = []
    for image in imgs:
        if(background is None and background_filenames is not None):
            bg_idx = np.random.randint(0, len(background_filenames))
            background = cv2.imread(background_filenames[bg_idx])
    
        if (len(image.shape) == 2):
            image = np.expand_dims(image, axis=2)
        
        if(image.shape[2] == 4):
            image = transparentOverlay(image, background, remove_mask=remove_mask)
        
        if(crop_percent is not None):
            image = cropAndResize(image, img_size, crop_percent)
        else:
            if(type(background) in [int, float]):
                padColor = background
            else:
                padColor = 255.0
            image = resizeAndPad(image, img_size, padColor=padColor)
        
        image = image.astype(np.uint8)
        if(normalize_tensors):
            if(vgg_normalize):
                if(remove_mask):
                    image = vggToTensor(image[:,:,:3])
                else:
                    image = torch.cat([vggToTensor(image[:,:,:3]), 
                                       vggToTensor(image[:,:,3:], mean=0)])
            else:
                if(remove_mask):
                    image = normalize(to_tensor(image[:,:,:3]))
                else:
                    image = torch.cat([normalize(to_tensor(image[:,:,:3])), 
                                       to_tensor(image[:,:,3:])])
        p_imgs.append(image)
        
    if(normalize_tensors):
        p_imgs = torch.stack(p_imgs)

    return p_imgs

def resizeAndPad(img, size, padColor=255.0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def cropAndResize(img, size, crop_percent):

    h, w = img.shape[:2]
    sh, sw = size

    ch = crop_percent*h
    cw = crop_percent*w

    if(ch/sh > cw/sw):
        ch = cw*sh/sw
    else:
        cw = ch*sw/sh
    
    ch = int(ch)
    cw = int(cw)
    
    r0 = int(h/2-ch/2)
    r1 = r0 + ch
    c0 = int(w/2-cw/2)
    c1 = c0 + cw
    cropped_img = img[r0:r1, c0:c1]

    # interpolation method
    if ch > sh or cw > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    scaled_img = cv2.resize(cropped_img, (sw, sh), interpolation=interp)
    
    return scaled_img
    
def transparentOverlay(foreground, background=None, remove_mask = True, pos=(0,0),scale = 1):
    """
    :param foreground: transparent Image (BGRA)
    :param background: Input Color Background Image
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Overlayed image
    """
    
    if(scale != 1):
        foreground = cv2.resize(foreground, None,fx=scale,fy=scale)

    alpha = foreground[:,:,3:].astype(float)/255    
   
    if(background is None):
        background = 255.0

    if(type(background) in [int, float]):
        img = alpha*foreground[:,:,:3] + background*(1.0-alpha)
        if(not remove_mask):
            img = np.concatenate([img, foreground[:,:,3:]], axis=2)
    else:
        h,w,_ = foreground.shape
        rows,cols,_ = background.shape
        
        y0,x0 = pos[0],pos[1]
        y1 = min(y0+h, rows)
        x1 = min(x0+w, cols)
        img = background.copy()
        img[y0:y1,x0:x1,:] = alpha*foreground[:,:,:3] + (1.0-alpha)*background[y0:y1,x0:x1,:]
        if(not remove_mask):
            img = np.concatenate([img, np.zeros((rows,cols,1))], axis=2)
            img[y0:y1,x0:x1,3:] = foreground[:,:,3:]

    return img

