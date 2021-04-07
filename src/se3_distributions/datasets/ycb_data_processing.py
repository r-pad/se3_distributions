import numpy as np
import numpy.ma as ma

import quat_math as qm

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

def getBBox(rois, img_width=480, img_length=640):
    rmin = int(rois[3]) + 1
    rmax = int(rois[5]) - 1
    cmin = int(rois[2]) + 1
    cmax = int(rois[4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def getYCBGroundtruth(pose_meta, posecnn_meta, data_idx):
    posecnn_idx = posecnn_meta['rois'][data_idx, 1]
    pose_idx = np.where(pose_meta['cls_indexes'].flatten()==posecnn_idx)[0][0]
    mat_gt = pose_meta['poses'][:,:, pose_idx]
    R_gt = np.eye(4)
    R_gt[:3,:3] = mat_gt[:3,:3]
    q_gt = qm.quaternion_from_matrix(R_gt)
    t_gt = mat_gt[:3,3]
    return q_gt, t_gt


def preprocessPoseCNNMetaData(posecnn_meta, idx = 0, img_size=(480, 640)):
    label = np.array(posecnn_meta['labels'])
    rois = np.array(posecnn_meta['rois'])
    lst = rois[:, 1:2].flatten()
    if(idx > len(lst)):
        return None
    itemid = lst[idx]

    bbox = getBBox(rois[idx], img_size[0], img_size[1])
    mask = ma.getmaskarray(ma.masked_equal(label, itemid))
    object_label = itemid - 1

    return mask, bbox, object_label
