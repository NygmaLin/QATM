from __future__ import print_function, division
import matplotlib.pyplot as plt
import math
from sklearn.metrics import auc
import numpy as np
import cv2
import os, sys
import numpy as np
from skimage.measure import compare_ssim


int_ = lambda x: int(round(x))


def IoU( r1, r2 ):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1; y12 = y11 + h1
    x22 = x21 + w2; y22 = y21 + h2
    x_overlap = max(0, min(x12,x22) - max(x11,x21) )
    y_overlap = max(0, min(y12,y22) - max(y11,y21) )
    I = 1. * x_overlap * y_overlap
    U = (y12-y11)*(x12-x11) + (y22-y21)*(x22-x21) - I
    J = I/U
    return J


def evaluate_iou( rect_gt, rect_pred ):
    # score of iou
    score = [ IoU(i, j) for i, j in zip(rect_gt, rect_pred) ]
    return score


def compute_score( x, w, h ):
    # score of response strength
    k = np.ones( (h, w) )
    score = cv2.filter2D(x, -1, k)
    score[:, :w//2] = 0
    score[:, math.ceil(-w/2):] = 0
    score[:h//2, :] = 0
    score[math.ceil(-h/2):, :] = 0
    return score


def locate_bbox( a, w, h ):
    row = np.argmax( np.max(a, axis=1) )
    col = np.argmax( np.max(a, axis=0) )
    x = col - 1. * w / 2
    y = row - 1. * h / 2
    return x, y, w, h


def score2curve( score, thres_delta = 0.01 ):
    thres = np.linspace( 0, 1, int(1./thres_delta)+1 )
    success_num = []
    for th in thres:
        success_num.append( np.sum(score >= (th+1e-6)) )
    success_rate = np.array(success_num) / len(score)
    return thres, success_rate


def all_sample_iou( score_list, gt_list):
    num_samples = len(score_list)
    iou_list = []
    for idx in range(num_samples):
        score, image_gt = score_list[idx], gt_list[idx]
        w, h = image_gt[2:]
        pred_rect = locate_bbox( score, w, h )
        iou = IoU( image_gt, pred_rect )
        iou_list.append( iou )
    return iou_list


def plot_success_curve( iou_score, title='' ):
    thres, success_rate = score2curve( iou_score, thres_delta = 0.05 )
    auc_ = np.mean( success_rate[:-1] ) # this is same auc protocol as used in previous template matching papers #auc_ = auc( thres, success_rate ) # this is the actual auc
    plt.figure()
    plt.grid(True)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0, 1)
    plt.title(title + 'auc={}'.format(auc_))
    plt.plot( thres, success_rate )
    plt.show()


def preprocess(image, keep_dim=False):
    # edge detect
    # detected_edges = cv2.GaussianBlur(image, (3, 3), 0)
    # detected_edges = cv2.Canny(detected_edges, 20, 100, apertureSize=3)
    # if keep_dim:
    #     detected_edges = np.tile(detected_edges[:, :, np.newaxis], (1, 1, 3))

    # erase black
    detected_edges = erase_black(image)

    return detected_edges


def binary(image):
    # edge detect
    detected_edges = cv2.GaussianBlur(image, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, 20, 100, apertureSize=3)
    return detected_edges


def erase_black(template_ori):
    h_valid = np.where(np.sum(template_ori, axis=0) > 0)
    w_valid = np.where(np.sum(template_ori, axis=1) > 0)
    template = template_ori[h_valid[0][0]:h_valid[0][-1] + 1, w_valid[0][0]:w_valid[0][-1] + 1]
    return template


def get_mask(patch, res_map):
    patch = binary(patch)
    kernel = np.ones((5, 5), np.uint8)
    dila_patch = cv2.dilate(patch, kernel, iterations = 1)
    res = res_map.copy()
    res[dila_patch == 255] = 255
    mask = res.copy()
    mask[res > 200] = 0
    mask[res <= 200] = 255
    return mask


def compare_show(box, template_path, image_path, resize_scale, save_path):

    template_name = template_path.split('\\')[-1]
    image = cv2.imread(image_path)
    # match = image[int(box[0][1] * 1 / resize_scale):int(box[1][1] * 1 / resize_scale),
    #         int(box[0][0] * 1 / resize_scale):int(box[1][0] * 1 / resize_scale), :]
    # match = cv2.resize(cv2.imread(image_path), size_image_raw)[box[0][1]:box[1][1], box[0][0]:box[1][0], :]
    template = cv2.imread(template_path)
    template = erase_black(template)

    # if match.shape != template.shape:
    #     h, w = template.shape
    #     match = cv2.resize(match, (w, h))
    search_region = image[max(0, int(box[0][1] * 1 / resize_scale) - 200):int(box[1][1] * 1 / resize_scale) + 200,
                    max(0, int(box[0][0] * 1 / resize_scale) - 200):int(box[1][0] * 1 / resize_scale) + 200, :]
    # import pdb
    # pdb.set_trace()
    match, min_val, res_map, template, s = temp_match(search_region, template)
    # s, res_map = compare_ssim(template, match,  win_size=3, full=True, gaussian_weights=False)
    res_path = r'E:\AIschool\QATM_pytorch_bak\save_path\part3\res'
    cv2.imwrite(os.path.join(res_path, template_name), res_map)

    mask = get_mask(match, res_map)
    merge = np.concatenate([match, template], axis=1)
    mask_path = r'E:\AIschool\QATM_pytorch_bak\save_path\part3\mask'

    cv2.imwrite(os.path.join(save_path, template_name), merge)
    cv2.imwrite(os.path.join(mask_path, template_name), mask)


def temp_match(img, template):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    h, w = template.shape[:2]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    patch = img_gray[min_loc[1]:min_loc[1] + h, min_loc[0]:min_loc[0] + w]
    (s, res_map) = compare_ssim(template, patch, win_size=3, full=True, gaussian_weights=False)
    return patch, min_val, res_map * 255, template, s

