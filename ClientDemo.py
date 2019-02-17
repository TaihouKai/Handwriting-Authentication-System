# coding: utf-8

import cv2
import time
import numpy as np

from extractor import HarrisSIFT
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


    def run(self, img, scale=0.3):
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
        dsts = list(map(lambda x: self.extractor.visualize(x, scale=1), img_batch))


        cv2.imshow('cropped', np.hstack(img_batch).astype(np.uint8))
        cv2.imshow('dst img', np.hstack(dsts).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



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

        e = HarrisSIFT.HarrisSIFT()

        client = HandWritingAuthClient(d, e, c)
        client.run(cv2.imread('samples/digit_data/standard_b.png')[:,:,:3], scale=1)


