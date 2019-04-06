# coding: utf-8


"""
HWAT Project
Copyright 2019
"""

import cv2
import time
import numpy as np

from extractor import *
from detector import *
from classifier import *
from util import *



class HandWritingAuthInstance():
    """ Handwriting Authentication Client
    """

    def __init__(self, detector, extractor):
        self.detector = detector
        self.extractor = extractor
        #   list of ndarray : [box_num] [kp_num, 2]
        self.kp_model = []
        #   list of ndarray : [box_num] [kp_num, feat_dim]
        self.feat_model = []

    def preproc(self, img, norm_height=64):
        #   Pre-processing
        simg = preprocessing.size_align(img, height=norm_height)
        bimg = preprocessing.binary(simg,
                                    cv2.GaussianBlur,
                                    (7, 7),
                                    threshold=0.75)
        return simg, bimg

    def register(self, imglist, min_poi=6):
        """
        Registration process
        If the PoI is smaller than `min_poi`, return False
        :param imglist:     list of images
        :param min_poi:     minimum PoI number
        :return:
        """
        reg_ratio, reg_kp, reg_feat = None, None, None
        for img in imglist:
            simg, bimg = self.preproc(img)
            #   Detection
            bboxes = self.detector.run(bimg, nms_thresh=0.1)

            #   Cropping & Padding (build batch)
            records = preprocessing.crop_img(bimg, bboxes)
            _, cimgs = list(map(list, zip(*records)))

            for img in cimgs:
                self.extractor.run(img)
        return reg_ratio, reg_kp, reg_feat

    def authenticate(self, img):
        """
        Authentication process
        :param img:
        :return:
        """


if __name__ == '__main__':
    d = ContourBox.ContourBox()

    e = SIFT.SIFT()
    test = CrossClassTest.CrossClassTest(base_dir='samples/digit_data')

    client = HandWritingAuthInstance(d, e)
    features = client.register(test.classes[0], min_poi=6)
