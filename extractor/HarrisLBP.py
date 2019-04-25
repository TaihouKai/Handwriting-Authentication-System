# coding: utf-8

"""
Fangrui Liu     mpskex@github   mpskex@163.com
Department of Computer Science and Technology
Faculty of Information
Beijing University of Technology
Copyright 2019
"""

import cv2
import sys
import time
import numpy as np
sys.path.append('..')

import util.cord_convert
import util.preprocessing

class HarrisLBP():
    """ Using Harris to extract possible corner
        And Using SIFT to encoding the feature
    """
    def __init__(self, ksize=2, block_size=3, k=0.16, blur=cv2.GaussianBlur):
        self.ksize = ksize
        self.block_size = block_size
        self.k = k
        self.blur = blur
        self.name = 'HarrisLBP'
    
    def run(self, img):
        """ Input must be gray image
        """
        feat = None
        dst = cv2.cornerHarris(img, self.block_size, self.ksize, self.k)
        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        kp = np.where(dst>0.01*dst.max())
        kp = np.transpose(np.array(kp), [1,0])
        return kp, feat
    
    def visualize(self, img, kp, color=(0, 0, 255)):
        kp = kp.tolist()
        # result is dilated for marking the corners, not important

        # Threshold for an optimal value, it may vary depending on the image.
        if len(img.shape) == 2:
            img = np.tile(np.expand_dims(img, axis=-1), [1, 1, 3])
        for k in kp:
            cv2.circle(img, tuple(k)[::-1], 5, color, 2)
        return img



if __name__ == '__main__':
    from util import CrossClassTest
    from util import preprocessing

    height=128
    def preproc(img):
        img = preprocessing.size_align(img, height=height)
        return preprocessing.binary(img, cv2.GaussianBlur, (7, 7), threshold=0.75)

    test = CrossClassTest.CrossClassTest()
    test.batch_process(preproc)

    n = HarrisLBP()
    print('Loaded model')
    t = time.time()
    ret_img = []
    num = 2
    for i, idx in zip(test.classes[num], range(len(test.classes[num]))):
        rimg = n.visualize(i).astype(np.uint8)
        cv2.imwrite('Harris_'+str(idx)+'.jpg', rimg)
        # cv2.imshow('Harris_'+str(idx)+'.jpg', rimg)

    cv2.waitKey()
    cv2.destroyAllWindows()
