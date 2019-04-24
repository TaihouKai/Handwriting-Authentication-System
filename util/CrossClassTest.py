# coding: utf-8
import cv2
import numpy as np

"""
Fangrui Liu     mpskex@github   mpskex@163.com
Department of Computer Science and Technology
Faculty of Information
Beijing University of Technology 
Copyright 2019
"""

class CrossClassTest:
    """
    Unit-test:  estimator test
    """
    def __init__(self, classes=None, base_dir='../samples/digit_data'):
        """
        Build up the test samples
        :param classes: nested list of images, [class]x[images]
        """
        self.base_dir = base_dir
        if classes is None:
            img_lfr = []
            img_standard = []
            img_wpf = []
            img_lfr.append(cv2.imread(self.base_dir + '/lfr_a.png', 0))
            img_lfr.append(cv2.imread(self.base_dir + '/lfr_b.png', 0))
            img_lfr.append(cv2.imread(self.base_dir + '/lfr_c.png', 0))
            img_standard.append(cv2.imread(self.base_dir + '/standard_a.png', 0))
            img_standard.append(cv2.imread(self.base_dir + '/standard_b.png', 0))
            img_wpf.append(cv2.imread(self.base_dir + '/wpf_a.png', 0))
            img_wpf.append(cv2.imread(self.base_dir + '/wpf_b.png', 0))
            img_wpf.append(cv2.imread(self.base_dir + '/wpf_c.png', 0))
            img_wpf.append(cv2.imread(self.base_dir + '/wpf_d.png', 0))
            img_wpf.append(cv2.imread(self.base_dir + '/wpf_e.png', 0))
            self.classes = [img_lfr, img_standard, img_wpf]
        else:
            self.classes = classes
        #   build test samples
        self.test_pair_same = []
        for _cls, _cls_idx in zip(self.classes, range(len(self.classes))):
            for a in range(len(_cls)):
                for b in range(len(_cls)):
                    if a < b:
                        self.test_pair_same.append([(_cls_idx, a), (_cls_idx, b)])
        self.test_pair_cross = []
        for _cls_a, _cls_a_idx in zip(self.classes, range(len(self.classes))):
            for _cls_b, _cls_b_idx in zip(self.classes, range(len(self.classes))):
                for a in range(len(_cls_a)):
                    for b in range(len(_cls_b)):
                        self.test_pair_cross.append([(_cls_a_idx, a), (_cls_b_idx, b)])
        print("Finished building test pairs!")

    def batch_process(self, batch_fn):
        self.classes = list(map(lambda cls: list(map(lambda data: batch_fn(data), cls)), self.classes))
        print('Batch processed!')

    def cross_test(self, test_fn, summ_fn):
        """
        test over samples
        :param test_fn: test function which generate data to summary function
                        NOTE:   test function should only consume two positional args
                                and a optional name input
        :param summ_fn: summary function that accepts all data (list) and gives a scalar

        :return summ_same:
        :return summ_cross:s
        """
        ret_dict = {}
        for pairs, name in zip([self.test_pair_same, self.test_pair_cross], ['same', 'cross']):
            ret_dict[name] = []
            for idx in pairs:
                ret_dict[name].append(test_fn(self.classes[idx[0][0]][idx[0][1]], self.classes[idx[1][0]][idx[1][1]],
                                              name=name+'_'+str(idx)))
            ret_dict[name] = summ_fn(ret_dict[name])
        return ret_dict['same'], ret_dict['cross']