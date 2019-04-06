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


def tlwh2tlbr(bbox, tolist=False):
    """
    :param bbox:
    :param tolist:
    :return:
    """
    x1, y1, w, h = np.split(bbox, 4, axis=-1)
    x2 = x1 + w
    y2 = y1 + h
    bbox = np.concatenate([x1, y1, x2, y2], axis=-1)
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