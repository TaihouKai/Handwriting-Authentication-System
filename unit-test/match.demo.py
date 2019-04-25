# coding: utf-8

"""
HWAT Project
Copyright 2019
"""


import sys
import cv2
import numpy as np

sys.path.append('..')
from ClientModel import HandWritingAuthInstance as hwai
from classifier import *
from extractor import *
from estimator import *
from detector import *

from util import *

colors = [(196, 203, 128), (136, 150, 0), (64, 77, 0),
          (201, 230, 200), (132, 199, 129), (71, 160, 67), (32, 94, 27),
          (130, 224, 255), (7, 193, 255), (0, 160, 255), (0, 111, 255),
          (220, 216, 207), (174, 164, 144), (139, 125, 96), (100, 90, 69),
          (252, 229, 179), (247, 195, 79), (229, 155, 3), (155, 87, 1),
          (231, 190, 225), (200, 104, 186), (176, 39, 156), (162, 31, 123),
          (210, 205, 255), (115, 115, 229), (80, 83, 239), (40, 40, 198),
          ]

e = HarrisLBP.HarrisLBP(ksize=3, block_size=2, k=0.114514)
d = ContourBox.ContourBox()


def test_match(ins, test_img_a, test_img_b, cross_pad=200):
    """
    Return match image
    The image is aligned to max(shape_a, shape_b) using np.pad & np.hstack
    ---------------------------
    | test_img_a | test_img_b |
    ---------------------------
    ^^^
    |||     order like this
    :param ins:             model instance
    :param test_img_a:      test image a
    :param test_img_b:      test image b
    :return:                visualized match image
    """
    #   extract key-points
    bimg_a, bboxes_a, ratios_a, kps_a, feats_a = ins.extract(test_img_a)
    bimg_b, bboxes_b, ratios_b, kps_b, feats_b = ins.extract(test_img_b)
    ratio_match = PointMatch.ratio_match(ratios_a, ratios_b, method='auth')
    print(ratio_match)
    #   parse internal var
    obboxes_a = bboxes_a
    obboxes_b = bboxes_b
    bboxes_a = cord_convert.tlbr_rev(cord_convert.tlbr2tlwh(bboxes_a))
    bboxes_b = cord_convert.tlbr_rev(cord_convert.tlbr2tlwh(bboxes_b))

    #   collect key-point pairs
    #   kp_pairs has shape of [bboxes, matches, (pair_a, pair_b), (y, x)]
    kp_pairs = []
    for match in ratio_match.tolist():
        single_pairs = []
        kps_match = PointMatch.keypoints_match(kps_a[match[0]], kps_b[match[1]], match_threshold=0.5)
        ka = cord_convert.denorm_point(kps_a[match[0]], bboxes_a[match[0], 2:4])
        kb = cord_convert.denorm_point(kps_b[match[1]], bboxes_b[match[1], 2:4])
        for km in kps_match.tolist():
            pair_a = ka[km[0]] + bboxes_a[match[0], :2]
            pair_b = kb[km[1]] + bboxes_b[match[1], :2] + np.array([0, bimg_a.shape[1]])
            single_pairs.append([pair_a.astype(np.int), pair_b.astype(np.int)])
        kp_pairs.append(single_pairs)

    #   visualize
    bimg_a_pad = np.pad(bimg_a, [[0, cross_pad], [0, 0]], mode='constant', constant_values=255)
    bimg_b_pad = np.pad(bimg_b, [[cross_pad, 0], [0, 0]], mode='constant', constant_values=255)
    stacked = np.tile(np.expand_dims(np.hstack([bimg_a_pad, bimg_b_pad]), axis=-1), [1, 1, 3])
    for pairs, kpidx in zip(kp_pairs, range(len(kp_pairs))):
        for p, pidx in zip(pairs, range(len(pairs))):
            #   draw circles and lines
            cv2.circle(stacked, tuple(p[0])[::-1],
                       radius=4, thickness=2, color=colors[pidx%len(colors)])
            cv2.circle(stacked, tuple(p[1]+np.array([cross_pad, 0]))[::-1],
                       radius=4, thickness=2, color=colors[pidx%len(colors)])
            cv2.line(stacked, tuple(p[0])[::-1], tuple(p[1]+np.array([cross_pad, 0]))[::-1],
                     color=colors[::-1][pidx%len(colors)], thickness=1)
        cv2.rectangle(stacked,
                      tuple(obboxes_a[ratio_match[kpidx][0], :2]),
                      tuple(obboxes_a[ratio_match[kpidx][0], 2:]),
                      color=colors[(kpidx-2)%len(colors)])
        cv2.rectangle(stacked,
                      tuple(obboxes_b[ratio_match[kpidx][1], :2]+np.array([bimg_a_pad.shape[1], 0])+np.array([0, cross_pad])),
                      tuple(obboxes_b[ratio_match[kpidx][1], 2:]+np.array([bimg_a_pad.shape[1], 0])+np.array([0, cross_pad])),
                      color=colors[(kpidx-2)%len(colors)])
    cv2.imwrite('save.jpg', stacked)

if __name__ == '__main__':

    test = CrossClassTest.CrossClassTest()

    test_img_a = test.classes[1][0]
    test_img_b = test.classes[1][1]
    ins = hwai(d, 
               e,
               debug=True)
    test_match(ins, test_img_a, test_img_b)
