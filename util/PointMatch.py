# coding: utf-8
import numpy as np
import sys
sys.path.append('..')
from util import nms
from util import cord_convert
"""
Fangrui Liu     mpskex@github   mpskex@163.com
Department of Computer Science and Technology
Faculty of Information
Beijing University of Technology 
Copyright 2019
"""


def ratio_match(box_reg, ratios_auth, std_err=1e-2):
    """
    Match bounding box ratios.
    We consider the bounding box as a sequential list.
    We need every ratios in `ration_reg` need to match a ratio_auth
    TODO:   (`std_err` need to be settled by the standard error among all samples.
    :param ratios_reg:  registered ratios ( list of float )
    :param ratios_auth: ratios which need authentication (list of float)
    :param std_err:     if exceed, give miss matched
    :return:            False, None for a fail case
                        True, Matched_index for a success case
    """
    def sgrtomtch(r_a, r_b):
        """
        single ratio match
        :param r_a:
        :param r_b:
        :return:    False
        """
        r_a, r_b = r_a - 1, r_b - 1
        #   same orientation
        if r_a * r_b < 0:
            return False
        if r_a - r_b <= std_err:
            return True
        else:
            return False

    idx_offset = 0
    match = []
    for r_idx in range(len(ratios_reg)):
        # out of range
        if idx_offset + r_idx >= len(ratios_auth):
            return False, None
        while not sgrtomtch(ratios_reg[r_idx], ratios_auth[r_idx+idx_offset]):
            idx_offset += 1
        match.append(r_idx+idx_offset)
    #   every bounding box should present in ratios_auth
    if len(match) != len(ratios_reg):
        return False, None
    else:
        return True, match

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    b_a = np.expand_dims(box_a, axis=1)
    b_b = np.expand_dims(box_b, axis=0)
    b_a = np.tile(b_a, [1, B, 1])
    b_b = np.tile(b_b, [A, 1, 1])

    def intersect(b_a, b_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        max_xy = np.minimum(b_a[:, :, 2:], b_b[:, :, 2:])
        min_xy = np.maximum(b_a[:, :, :2], b_b[:, :, :2])
        inter = np.maximum((max_xy - min_xy), 0)
        return inter[:, :, 0] * inter[:, :, 1]

    inter = intersect(b_a, b_b)
    area_a = ((b_a[:, :, 2]-b_a[:, :, 0]) * (b_a[:, :, 3]-b_a[:, :, 1]))
    area_b = ((b_b[:, :, 2]-b_b[:, :, 0]) * (b_b[:, :, 3]-b_b[:, :, 1]))
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def keypoints_match(keypoints_a, keypoints_b, match_threshold=0.5, inter_threshold=0.5):
    """
    Match sets of key-points
    :param keypoints_a:     ndarray of shape [kp_num_a, 2]
    :param keypoints_b:     ndarray of shape [kp_num_b, 2]
    :param match_threshold: threshold of match intersection
    :param inter_threshold: threshold of self intersection
    :return match:          [matches, (ind())]

    """
    kp_a, kp_b = list(map(lambda kp: cord_convert.tlbr_rev(
        cord_convert.cwh2tlbr(
            np.concatenate([kp, 1*np.ones(list(kp.shape[:-1])+[2])], axis=-1))),
                          [keypoints_a, keypoints_b]))
    kp_a = nms.non_max_suppression_fast(kp_a, inter_threshold)
    kp_b = nms.non_max_suppression_fast(kp_b, inter_threshold)
    print(kp_a)
    print(kp_b)
    j = jaccard(kp_a, kp_b)
    match = np.array(np.where(j > match_threshold))
    return match



if __name__ == '__main__':
    print(keypoints_match(np.array([[1,2],[2,4]]), np.array([[-2,-1], [2,-2], [3, 1]])))