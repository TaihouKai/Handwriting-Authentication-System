# coding: utf-8
import cv2
import numpy as np


class CrossClassTest:
    """
    Unit-test:  estimator test
    """
    def __init__(self, classes=None):
        """
        Build up the test samples
        :param classes: nested list of images, [class]x[images]
        """
        if classes is None:
            img_lfr = []
            img_standard = []
            img_wpf = []
            img_lfr.append(cv2.imread('../samples/digit_data/lfr_a.png', 0))
            img_lfr.append(cv2.imread('../samples/digit_data/lfr_b.png', 0))
            img_lfr.append(cv2.imread('../samples/digit_data/lfr_c.png', 0))
            img_standard.append(cv2.imread('../samples/digit_data/standard_a.png', 0))
            img_standard.append(cv2.imread('../samples/digit_data/standard_b.png', 0))
            img_wpf.append(cv2.imread('../samples/digit_data/wpf_a.png', 0))
            img_wpf.append(cv2.imread('../samples/digit_data/wpf_b.png', 0))
            img_wpf.append(cv2.imread('../samples/digit_data/wpf_c.png', 0))
            self.classes = [img_lfr, img_standard, img_wpf]
        else:
            self.classes = classes
        #   build test samples
        self.test_pair_same = []
        for _cls in self.classes:
            for a in _cls:
                for b in _cls:
                    if a is not b:
                        self.test_pair_same.append([a, b])
        self.test_pair_cross = []
        for _cls_a in self.classes:
            for _cls_b in self.classes:
                for a in _cls_a:
                    for b in _cls_b:
                        self.test_pair_cross.append([a, b])
        print("Finished building test pairs!")

    def batch_process(self, batch_fn):
        self.classes = list(map(lambda cls: list(map(lambda data: batch_fn(data), cls)), self.classes))

    def test(self, test_fn, summ_fn):
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
            for pair, idx in zip(pairs, range(len(pairs))):
                ret_dict[name].append(test_fn(pair[0], pair[1], name=name+'_'+str(idx)))
            ret_dict[name] = summ_fn(ret_dict[name])
        return ret_dict['same'], ret_dict['cross']