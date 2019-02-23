# coding: utf-8


"""
HWAT Project
Copyright 2019
"""

import cv2
import time
import numpy as np

from extractor import SIFT
from detector import ContourBox
from classifier import AlexNet

from util import preprocessing, cord_convert


class HandWritingAuthClient():
    """ Handwriting Authentication Client
    """
    def __init__(self, detector, extractor, classifier):
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
    
    def preproc(self, img, scale=0.3):
        #   Pre-processing
        simg = preprocessing.scale(img, factor=scale)
        bimg = preprocessing.binary(simg,
                                    cv2.GaussianBlur,
                                    (7,7),
                                    threshold=0.7)
        return simg, bimg


    def run(self, img, scale=0.3, debug=False):
        simg, bimg = self.preproc(img, scale=scale)
        #   Detection
        bboxes = self.detector.run(bimg, nms_thresh=0.1)

        #   Cropping & Padding (build batch)
        records = preprocessing.crop_img(bimg, bboxes)
        records = np.array(records)
        print(records.shape)
        img_batch = list(map(lambda x: 
                                    preprocessing.mold_image(x, 
                                                            (28, 28)),
                            records[:, 1]))

        #   Classification
        pred = self.classifier.predict(img_batch)
        print(pred)

        #   Feature Extraction
        des = list(map(lambda x: self.extractor.run(
                                        np.tile(np.expand_dims(x, axis=-1),[1,1,3])), 
                                    img_batch))

        #   In registration Process:
        #       We match the feature in each class
        #       Then we count the presence of the similar feature
        #       Keep the common feature in a person's writing
        #   In Verification Process:
        #       We extract the feature by class
        #       Then we send the feature to the server
        #       Use Homomorphic Crypto (If Applicable) to match the feature
        #   
        #   Hint:
        #       We can use SVM to give the confidence of verification

        #   build feature detection result
        features = {}
        for idx in range(len(pred)):
            #   dict : '<class_id>' : feature array in shape of (num_detection, 128)
            features[str(pred[idx])] = des[idx][1]
        if debug:
            cv2.imshow('cropped', np.hstack(img_batch).astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return features



if __name__ == '__main__':
        config = AlexNet.AlexNetConfig()
        config.read('config/AlexNet.cfg')
        config.bound(train_flag=False)
        config.print_config()

        c = AlexNet.AlexNet(config=config, 
                    training=False,
                    load_pretrained=True,
                    pretrained_model='model/AlexNet.npy'
                    )
        c.BuildModel()

        d = ContourBox.ContourBox()

        e = SIFT.SIFT()

        client = HandWritingAuthClient(d, e, c)
        features = client.run(cv2.imread('samples/digit_data/standard_b.png')[:,:,:3], scale=1)
        for key in features.keys():
            if features[key] is None:
                print(key, ":\tNone")
            else:
                print(key, ":\t", features[key].shape)
