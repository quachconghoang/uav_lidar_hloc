import cv2 as cv
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch
import ot

from third_party.SuperGluePretrainedNetwork.models.matching import Matching
from third_party.SuperGluePretrainedNetwork.models.utils import (make_matching_plot_fast, frame2tensor)
torch.set_grad_enabled(False)

def showRGB(img):
    plt.imshow(img)
    plt.show()

def showDepth(dimg, dmax = 30):
    plt.imshow(dimg, cmap='gray', vmin=0, vmax=dmax)
    plt.show()

def showNorm(mat, vmax=1.7):
    plt.imshow(mat, cmap='gray', vmin=0, vmax=vmax)
    plt.show()


def getDistanceMask(img_depth, thresh = 15.):
    return (np.abs(img_depth) < thresh).reshape(-1)

import torch

dnn_config = {
    'superpoint': {
        'nms_radius': 8,
        'keypoint_threshold': 0.05,
        # 'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    # 'superpoint': {
    #     'nms_radius': 4,
    #     'keypoint_threshold': 0.01,
    #     'max_keypoints': 2048
    # },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

dnn_device = 'cuda'



matching = Matching(dnn_config).eval().to(dnn_device)

def getSuperPoints_v2(src_gray):
    # src_gray_x2 = cv.resize(src_gray, (320, 240), interpolation=cv.INTER_AREA)
    # src_gray_x4 = cv.resize(src_gray, (160, 120), interpolation=cv.INTER_AREA)

    frame_tensor = frame2tensor(src_gray, dnn_device)
    dat = matching.superpoint({'image': frame_tensor})
    # keys = ['keypoints', 'scores', 'descriptors']
    pts = dat['keypoints'][0].cpu().numpy()
    scores = dat['scores'][0].cpu().numpy()
    desc = dat['descriptors'][0].cpu().numpy()
    return {
        'pts': pts,
        'scores': scores,
        'desc': desc
    }

def hungarianMatch(norm_cross, sorted=False):
    match_id = linear_sum_assignment(cost_matrix = norm_cross)
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    if sorted:
        idx_sorted = np.argsort(match_score)
        matches = matches[idx_sorted, :]

    return matches

def sinkhornMatch(norm_cross, norm_src, norm_tar, lambd=1e-1, iter=20):
    a, b = [], []
    for pt_his in norm_src:
        a.append(pt_his.mean() - 1)
    for pt_his in norm_tar:
        b.append(pt_his.mean() - 1)

    Gs = ot.sinkhorn(a, b, norm_cross, reg=lambd, numItermax=iter)
    matches = hungarianMatch(1 - Gs)
    return matches

def getAnchorPoints(cost_matrix , his_src, his_tar, sorting = False):
    Gs = ot.sinkhorn(his_src, his_tar, cost_matrix, reg=1e-1, numItermax=20)
    match_id = linear_sum_assignment(cost_matrix = Gs, maximize=True)
    match_score = Gs[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    if sorting:
        idx_sorted = np.argsort(-match_score) #high-to-low
        matches = matches[idx_sorted, :]

    theshold = Gs.max()*0.8
    keep = matches[:,2] > theshold
    matches = matches[keep]

    return matches, Gs