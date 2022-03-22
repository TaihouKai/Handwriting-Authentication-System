# coding: utf-8

"""
Fangrui Liu     mpskex@github   mpskex@163.com
Department of Computer Science and Technology
Faculty of Information
Beijing University of Technology
Copyright 2019
"""

import cv2
import numpy as np
from . import cord_convert

def binary(img, blur, kernel_size, threshold=0.7):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    #   Normalized to 0-1
    gray = normlize(gray, mode='spatial')
    if blur is not None:
        blur_img = blur(gray, kernel_size, 0)
    else:
        blur_img = gray
    blur_img[np.where(blur_img > threshold)] = 255
    blur_img[np.where(blur_img <= threshold)] = 0
    return blur_img.astype(np.uint8)

def scale(img, factor=0.5):
    return cv2.resize(img, (int(factor*img.shape[1]), int(factor*img.shape[0])))

def size_align(img, height=32):
    ratio = img.shape[1] / float(img.shape[0])
    size = (int(ratio*height), int(height))
    return cv2.resize(img, size)

def normlize(array, mode='dimensional', axis=None):
    """
    Normalize a N-D array
    :param array:   numpy ndarray
    :param mode:    'spatial' for 3D-array like images
                    'dimensional' to normalize regarding to every dimensions
    :return:
    """
    assert mode in ['spatial', 'dimensional']
    if mode == 'spatial':
        axis = (0, 1)
    elif mode == 'dimensional':
        if axis is not None:
            axis = tuple(axis)
    min = np.min(array, axis = axis)
    max = np.max(array, axis = axis)
    return (array - min) / float(max - min)




def create_crop_data(img, bbox):
    """
    create cropped image and data
    bbox in cwh format

    in storage we process the data in tlbr format
    """
    return (bbox, img[bbox[1]:bbox[3], bbox[0]:bbox[2]])


def crop_img(img, bboxes):
    """
    Crop image according to bounding boxes
    """
    return list(map(lambda x: create_crop_data(img, np.array(x)), list(bboxes)))

def decrop_img(img, bboxes, patches):
    """
    Patch those images back to images
    """
    # Threshold for an optimal value, it may vary depending on the image.
    if len(img.shape) == 2:
        img = np.tile(np.expand_dims(img, axis=-1), [1, 1, 3])
    for bbox, bidx in zip(bboxes, range(bboxes.shape[0])):
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = patches[bidx]
    return img

def mold_image(img, in_size, margin=0.7, padd_value=0, reversed=True):
    """
    Args:
        img         numpy 2darray
        in_size     tuple
    """
    #   resize
    h_scale = in_size[1] / float(img.shape[0])
    w_scale = in_size[0] / float(img.shape[1])
    if h_scale < 1 or w_scale < 1:
        scale = max(h_scale, w_scale)
        size = np.array(img.shape) * scale
        img = cv2.resize(img, tuple(size.astype(np.int)))
    else:
        scale = 1
    
    #   padding
    padd_x = img.shape[1] - in_size[0] 
    padd_y = img.shape[0] - in_size[1]
    if padd_x < 0:
        padd_x = 0
    if padd_y < 0:
        padd_y = 0
    padd_x += int(in_size[0] * margin)
    padd_y += int(in_size[1] * margin)
    padd = [[padd_y//2, padd_y//2], [padd_x//2, padd_x//2]]
    if len(img.shape) == 3:
        padd.append([0,0])
    if reversed:
        img = cv2.bitwise_not(img)
    img = np.pad(img, padd, 'constant', constant_values=padd_value)
    img = cv2.resize(img, in_size)
    return img
    
    
        