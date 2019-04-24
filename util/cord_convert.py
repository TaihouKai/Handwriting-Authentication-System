#coding: utf-8
import numpy as np

"""
Fangrui Liu     mpskex@github   mpskex@163.com
Department of Computer Science and Technology
Faculty of Information
Beijing University of Technology 
Copyright 2019
"""

def cwh2tlbr(bbox, tolist=False):
    """ Cx, Cy, W, H to TopLeft BottomRight
    """
    cx, cy, w, h = np.split(bbox, 4, axis=-1)
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    box = np.concatenate([x1, y1, x2, y2], axis=-1)
    if tolist:
        return box.tolist()
    else:
        return box


def tlwh2tlbr(bbox, tolist=False):
    """ Cx, Cy, W, H to TopLeft BottomRight
    """
    x, y, w, h = np.split(bbox, 4, axis=-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    box = np.concatenate([x1, y1, x2, y2], axis=-1)
    if tolist:
        return box.tolist()
    else:
        return box


def tlbr2cwh(bbox, tolist=False):
    """ Cx, Cy, W, H to TopLeft BottomRight
    """
    x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
    w = np.abs(x2 - x1)
    h = np.abs(y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    bbox = np.concatenate([cx, cy, w, h], axis=-1)
    if tolist:
        return bbox.tolist()
    else:
        return bbox


def tlbr_rev(bbox, tolist=False):
    """
    :param bbox:
    :param tolist:
    :return:
    """
    x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
    bbox = np.concatenate([y1, x1, y2, x2], axis=-1)
    if tolist:
        return bbox.tolist()
    else:
        return bbox


def tlbr2tlwh(bbox, tolist=False):
    """
    :param bbox:
    :param tolist:
    :return:
    """
    x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
    w = x2 - x1
    h = y2 - y1
    bbox = np.concatenate([x1, y1, w, h], axis=-1)
    if tolist:
        return bbox.tolist()
    else:
        return bbox


def norm_point(point, norm_shape, reversed_dim=True):
    """
    norm point to 0-1
    NOTE: the order of the point's dim is not changed after the function
    :param point:           ndarray of shape [num_point, num_dim]
    :param norm_shape:      ndarray of shape [num_dim]
    :param reversed_dim:    If the shape's dim has be reversed
    :return:
    """
    assert len(np.array(norm_shape).shape) == 1
    assert point.shape[-1] == np.array(norm_shape).shape[0]
    point = point.copy()
    for dim in range(np.array(norm_shape).shape[0]):
        point[:, dim] = point[:, dim] / norm_shape[::(int(reversed_dim) * 2 - 1)][dim]
    return point


def denorm_point(point, norm_shape, reversed_dim=True):
    """
    denorm point to 0-1
    NOTE: the order of the point's dim is not changed after the function
    :param point:           ndarray of shape [num_point, num_dim]
    :param norm_shape:      ndarray of shape [num_dim]
    :param reversed_dim:    If the shape's dim has be reversed
    :return:
    """
    assert len(np.array(norm_shape).shape) == 1
    assert point.shape[-1] == np.array(norm_shape).shape[0]
    point = point.copy()
    for dim in range(np.array(norm_shape).shape[0]):
        point[:, dim] = point[:, dim] * norm_shape[::(int(reversed_dim) * 2 - 1)][dim]
    return point


if __name__ == '__main__':
    a = norm_point(np.transpose(np.array([[12, 24], [8, 36]]), axes=[1, 0]), np.array([2, 4]), reversed_dim=True)
    b = denorm_point(a, np.array([2, 4]), reversed_dim=True)
    print(a)
    print(b)