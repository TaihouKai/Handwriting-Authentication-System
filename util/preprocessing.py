# coding: utf-8
import cv2
import numpy as np
from . import cord_convert

def binary(img, blur, kernel_size, threshold=0.7):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    #   Normalized to 0-1
    gray = (gray - np.amin(gray)) / (np.amax(gray) - np.amin(gray))
    if blur is not None:
        blur_img = blur(gray, kernel_size, 0)
    else:
        blur_img = gray
    blur_img[np.where(blur_img > threshold)] = 255
    blur_img[np.where(blur_img <= threshold)] = 0
    return blur_img.astype(np.uint8)

def scale(img, factor=0.5):
    return cv2.resize(img, (int(factor*img.shape[1]), int(factor*img.shape[0])))


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


def mold_image(img, in_size, margin=0.7, padd_value=0, reversed=True):
    """
    Args:
        img         numpy 2darray
        in_size     tuple
    """
    #   resize
    h_scale = in_size[1] / img.shape[0]
    w_scale = in_size[0] / img.shape[1]
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
    
    
        