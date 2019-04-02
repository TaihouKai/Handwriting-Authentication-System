# coding: utf-8
import cv2
import numpy as np
import sys
sys.path.append('..')

class NaiveEstimator():
    """
    Naive Estimator that gives similarity between two given samples
    """
    def __init__(self):
        """
        Constructor
        """
        pass

    def compare(self, matches):
        """
        Compute the similarity between two sets of features
        :param matches:     list of tuple of DMatch
        """
        distances = []
        for m, in matches:
            distances.append(abs(m.distance))
        distances = np.array(distances)
        #   We use the Mean & VAR to compute similarity
        mean = np.mean(distances)
        var = np.var(distances)
        return mean, var


if __name__ == '__main__':
    from extractor import SIFT
    import util.preprocessing
    from util import CrossClassTest

    sift = SIFT.SIFT()
    naive = NaiveEstimator()
    test = CrossClassTest.CrossClassTest()

    def test_fn(img_a, img_b, name='out'):
        good = sift.match(img_a, img_b, name=name)
        return naive.compare(good)

    def summ_fn(data):
        data = np.array(list(map(list, zip(*data))))
        return data

    def preproc(img):
        factor = 0.3
        img = cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)))
        return util.preprocessing.binary(img, cv2.GaussianBlur, (7, 7), threshold=0.75)

    test.batch_process(preproc)
    r_same, r_cross = test.test(test_fn, summ_fn)
    print(np.mean(r_same, axis=-1), np.var(r_same, axis=-1))
    print(np.mean(r_cross, axis=-1), np.var(r_cross, axis=-1))