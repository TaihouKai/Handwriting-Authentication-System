# coding: utf-8

"""
HWAT Project
Copyright 2019
"""

import cv2
import sys
import time
import numpy as np
sys.path.append('..')

import util.cord_convert
import util.preprocessing

class HarrisSIFT():
    """ Using Harris to extract possible corner
        And Using SIFT to encoding the feature
    """
    def __init__(self, ksize=2, block_size=3, k=0.16, blur=cv2.GaussianBlur):
        self.ksize = ksize
        self.block_size = block_size
        self.k = k
        self.blur = blur
    
    def run(self, img):
        """ Input must be gray image
        """
        dst = cv2.cornerHarris(img, self.ksize, self.block_size , self.k)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        return dst 
    
    def visualize(self, img, scale=0.3):
        img = util.preprocessing.binary(img, cv2.GaussianBlur, (5,5), 0.2)
        if scale != 1:
            img = util.preprocessing.scale(img, factor=scale)
        if len(img.shape) == 3:
            gray = util.preprocessing.binary(img, None, None, 0.7)
        else:
            gray = img
        gray = np.float32(gray)
        dst = self.run(gray)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        if len(img.shape) == 2:
            img = np.tile(np.expand_dims(img, axis=-1), [1,1,3])
        img[dst>0.01*dst.max()]=[0,0,255]
        return img



if __name__ == '__main__':
    n = HarrisSIFT()
    print('Loaded model')
    t = time.time()
    img, dst = n.visualize(cv2.imread('../samples/digit_data/standard_a.png')[:,:,:3], scale=1)
    print(time.time()-t)
    cv2.imshow('img', util.preprocessing.scale(img.astype(np.uint8), factor=1))
    cv2.imshow('dst', util.preprocessing.scale(dst.astype(np.uint8), factor=1))
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
