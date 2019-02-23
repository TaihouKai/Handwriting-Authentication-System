# coding: utf-8

"""
HWAT Project
Copyright 2019
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def print_keypoints(keypoints):
    """
    Args:
        keypoints:  consume list of keypoints
    """
    for kp in keypoints:
        print("cord:\t", kp.pt)
        print("size:\t", kp.size)
        print("angle:\t", kp.angle)
        print("resp:\t", kp.response)
        print("octav:\t", kp.octave)
        print("class:\t", kp.class_id)

class SIFT():
    """ SIFT is patented algorithm
        We need an alternative option for extractor
    """
    def __init__(self, debug=False):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.debug = debug

    def run(self, img):
        """ Accept RGB image
        """
        kp, des = self.sift.detectAndCompute(img, None)
        if self.debug:
            print(len(kp), des.shape)
        return kp, des


if __name__ == '__main__':
    img1 = cv2.imread('../samples/digit_data/standard_a.png', 0)          # queryImage
    img2 = cv2.imread('../samples/digit_data/lfr_a.png', 0) # trainImage

    sift = SIFT()
    kp1, des1 = sift.run(img1)
    kp2, des2 = sift.run(img2)

    print_keypoints(kp1[:2])

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = None
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)
    cv2.imwrite('feature.test.jpg', img3)
    plt.imshow(img3),plt.show()
    np.savetxt('feature.csv', des1)