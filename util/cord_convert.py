#coding: utf-8
import numpy as np


def cwh2tlbr(bbox, tolist=True):
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

def tlwh2tlbr(bbox, tolist=True):
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

def tlbr2cwh(bbox, tolist=True):
    """ Cx, Cy, W, H to TopLeft BottomRight
    """
    x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
    w = np.abs(x2 - x1)
    h = np.abs(y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    box = np.concatenate([cx, cy, w, h], axis=-1)
    if tolist:
        return box.tolist()
    else:
        return box

