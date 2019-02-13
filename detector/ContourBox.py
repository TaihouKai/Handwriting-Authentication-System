#coding: utf-8

import cv2
import time
import numpy as np
import sys
sys.path.append('..')

import util.nms
import util.cord_convert
import util.preprocessing

class ContourBox():
    """ Use Contour to draw boxes

    *   generate() 
    """
    def __init__(self, bin_threshold=0.7, kernel_size=7, blur=cv2.GaussianBlur):
        self.bin_threshold = bin_threshold
        self.blur = blur
        self.kernel_size = (kernel_size, kernel_size)


    def run(self, img):
        """ Input must be binary image (0 or 255) / gray image (0 ~ 255)
        """
        _, thresh = cv2.threshold(img, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        bboxes = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            bboxes.append(box)
        bboxes = np.array(bboxes)
        bboxes = util.cord_convert.tlwh2tlbr(bboxes, tolist=False)
        bboxes = util.nms.non_max_suppression_fast(bboxes, 0.05)
        return bboxes[1:]


    def generate(self, img):
        binary = util.preprocessing.binary(img, self.blur, self.kernel_size, threshold=self.bin_threshold)
        return self.run(binary).tolist()


    def visualize(self, img, scale=0.5):
        img = util.preprocessing.scale(img, factor=scale)
        bboxes = self.generate(img)
        print(len(bboxes))
        for box in bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        return img


if __name__ == '__main__':
    n = ContourBox(kernel_size=7)
    print('Loaded model')
    t = time.time()
    retimg = n.visualize(cv2.imread('../samples/digit_data/lfr_a.png')[:,:,:3], scale=0.3)
    print(time.time()-t)
    cv2.imshow('contour', retimg.astype(np.uint8))
    cv2.waitKey(0)
