#coding: utf-8

import cv2
import time
import numpy as np
import sys
sys.path.append('..')

import util.nms
import util.cord_convert

class ContourBox():
    """ Use Contour to draw boxes
    """
    def __init__(self, bin_threshold=0.7, kernel_size=7, blur=cv2.GaussianBlur):
        self.bin_threshold = bin_threshold
        self.blur = blur
        self.kernel_size = (kernel_size, kernel_size)
    
    def binary(self, img, blur, kernel_size, threshold=0.4):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #   Normalized to 0-1
        gray = (gray - np.amin(gray)) / (np.amax(gray) - np.amin(gray))
        blur_img = blur(gray, kernel_size, 0)
        blur_img[np.where(blur_img > threshold)] = 255
        blur_img[np.where(blur_img <= threshold)] = 0
        print('blur_img has shape of ', blur_img.shape)
        return blur_img.astype(np.uint8)
    
    def scale(self, img, factor=0.5):
        return cv2.resize(img, (int(factor*img.shape[1]), int(factor*img.shape[0])))

    def visualize(self, img, scale=0.5):
        img = self.scale(img, factor=scale)
        binary = self.binary(img, self.blur, self.kernel_size, threshold=self.bin_threshold)
        _, thresh = cv2.threshold(binary, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        bboxes = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            bboxes.append(box)
        bboxes = np.array(bboxes)
        bboxes = util.cord_convert.tlwh2tlbr(bboxes, tolist=False)
        bboxes = util.nms.non_max_suppression_fast(bboxes, 0.1)
        bboxes = bboxes.tolist()
        print(len(bboxes))
        for box in bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        return img


if __name__ == '__main__':
    n = ContourBox(kernel_size=7)
    print('Loaded model')
    t = time.time()
    retimg = n.visualize(cv2.imread('../src/digit_data/lfr_a.png')[:,:,:3], scale=0.3)
    print(time.time()-t)
    cv2.imshow('contour', retimg.astype(np.uint8))
    cv2.waitKey(0)
