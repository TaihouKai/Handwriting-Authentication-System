# coding: utf-8

"""
HWAT Project
Copyright 2019
"""

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from util.CrossClassTest import CrossClassTest
from util import preprocessing
from extractor.SIFT import SIFT
from estimator.NaiveEstimator import NaiveEstimator

if __name__ == '__main__':
    naive = NaiveEstimator()
    sift = SIFT()
    kernel_size_range = range(3, 11, 2)
    factor_range = np.arange(0.1, 1, 0.1).tolist()
    factor_graph = []
    for factor in factor_range:
        def test_fn(img_a, img_b, name='out'):
            good = sift.match(img_a, img_b, name=name)
            return naive.compare(good)

        def summ_fn(data):
            data = list(map(np.array, zip(*data)))
            data = list(map(lambda x: np.delete(x, np.where(x==0)), data))
            return data

        def preproc(img):
            img = cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)))
            return preprocessing.binary(img, cv2.GaussianBlur, (7, 7), threshold=0.8)

        naive = NaiveEstimator()
        sift = SIFT()
        test = CrossClassTest()
        test.batch_process(preproc)
        r_same, r_cross = test.cross_test(test_fn, summ_fn)
        #   (mean, var) x (same, cross)
        result = list(map(lambda fn: list(map(lambda data: fn(data), [r_same, r_cross])),
                          [lambda x: list(map(lambda _x: np.mean(_x, axis=-1), x)),
                           lambda x: list(map(lambda _x: np.var(_x, axis=-1), x))]))
        factor_graph.append(result)
        print("Collected result of plot " + str(factor))
        del naive, test, sift
    #   [point, 2, 2, result_dim]
    factor_graph = np.array(factor_graph)
    plt.figure()
    for subsubidx in range(factor_graph.shape[3]):
        for _str, idx in zip(['mean', 'var'], range(factor_graph.shape[1])):
            plt.subplot(factor_graph.shape[0], factor_graph.shape[1], subsubidx*factor_graph.shape[3] + idx + 1)
            for _substr, color, subidx in zip(['same', 'cross'], ['red', 'blue'], range(factor_graph.shape[2])):
                #   compare between same & cross
                y = factor_graph[:, idx, subidx, subsubidx]
                plt.plot(factor_range, y, color=color)
    plt.show()
