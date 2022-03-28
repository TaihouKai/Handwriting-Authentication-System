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
# Cannot install tensorflow in py2 env using pip
#from classifier import *
from util import *


class HandWritingAuthInstance():
    """ Handwriting Authentication Client
    """

    def __init__(self, detector, extractor, debug=False):
        self.debug = debug
        self.detector = detector
        self.extractor = extractor
        #   list of ndarray : [box_num] [kp_num, 2]
        #   Key-point spatial model
        self.kp_model = []
        #   list of ndarray : [box_num] [kp_num, feat_dim]
        #   Feature descriptor model
        self.feat_model = []
        #   list of float
        #   Bounding box ratio sequence model
        self.ratio_model = []

    def preproc(self, img, norm_height=128):
        #   Pre-processing
        simg = preprocessing.size_align(img, height=norm_height)
        bimg = preprocessing.binary(simg,
                                    cv2.GaussianBlur,
                                    (7, 7),
                                    threshold=0.1)
        return simg, bimg

    def get_area(self, bboxes):
        """
        receive TLBR bboxes
        :param bboxes:
        :return:
        """
        _, _, w, h = np.split(cord_convert.tlbr2tlwh(bboxes), 4, axis=-1)
        area = w * h
        return np.squeeze(area, axis=-1)

    def extract(self, img):
        simg, bimg = self.preproc(img, 256)


        #   Get the Bounding Boxes
        bboxes = self.detector.run(bimg, nms_thresh=0.1)
        # bimg_ = np.stack([bimg] * 3, axis=-1).astype(np.uint8) 
        # print(bboxes, bimg_.shape)
        # for n in list(bboxes):
        #     bimg_ = cv2.rectangle(bimg_, n[:2], n[2:], (0, 255, 0), 2)
        # cv2.imshow('test', bimg_)
        # cv2.waitKey(0)
        bindx = np.argsort(np.array(bboxes)[:, 0])
        bboxes = bboxes[bindx]
        areas = self.get_area(bboxes)
        bboxes = np.delete(bboxes, np.where(areas < 100), axis=0)
        #   Cropping & Padding (build batch)
        records = preprocessing.crop_img(bimg, bboxes)
        #   Transpose the list into arg dimension
        _, cimgs = list(map(list, zip(*records)))

        #   Collect result from every bounding boxes
        ratios, kps, feats = [], [], []
        for cidx in range(len(records)):
            #   Get the result from extractor (SIFT or Harris+LBP)
            result = self.extractor.run(cimgs[cidx])
            if self.extractor.name == 'SIFT':
                #   SIFT is a OpenCV Built-in function and it's non-free
                #   returns a Python-unfriendly type -- list of <DMatch>
                #   Maybe we need to do this in Python code
                #   or write a parser for this
                #   TODO: SIFT Feature Registration
                kp, feat = None, None
                raise NotImplementedError
            elif self.extractor.name == 'HarrisLBP':
                #   For Harris+LBP, it's easy for us to extract those corners
                #   By using those key-points match functions, we can collect
                #   and calculate average position of PoI.
                kp, feat = result
            else:
                raise ValueError('Invalid extractor!')
            kps.append(cord_convert.norm_point(
                kp.astype(np.float32), cimgs[cidx].shape, reversed_dim=True))
            feats.append(feat)
            wh = cord_convert.tlbr2cwh(bboxes[cidx])[2:]
            ratios.append(wh[1]/float(wh[0]))
        return bimg, bboxes, ratios, kps, feats

    def register(self, imglist, min_poi=6, update_weight=0.3):
        """
        Registration process
        If the PoI is smaller than `min_poi`, return False
                     `ratio_match`                    `kp_match`
        Ratio match ===============> Key-point match ============> feature match

        *   count the matched key-point and calculate the mean one
        *   give the confident ratio estimation by grab top-k result

        :param imglist:         list of images
        :param min_poi:         minimum PoI number
        :return:
        """
        reg_ratio, reg_kp, reg_feat = None, None, None
        clct_ratio, clct_kp, clct_feat = [], [], []
        clct_idx = 0
        for img in imglist:
            _, _, ratios, kps, feats = self.extract(img)
            list(map(lambda l, elem: l.append(elem), [
                 clct_ratio, clct_kp, clct_feat], [ratios, kps, feats]))

        #   Summarize the bio auth info
        assert len(clct_kp) == len(clct_feat) == len(clct_ratio)
        #   To simplify the process, we only accept a set of pictures
        #   We can do more than this, online-registration process actually is available
        for single_kps, single_ratios, single_feats in zip(clct_kp, clct_ratio, clct_feat):
            single_ratios = np.array(single_ratios)
            #   Register ratios
            if reg_ratio is None:
                #   (ratio, weight)
                reg_ratio = np.stack(
                    [single_ratios, np.zeros([single_ratios.shape[0]])], -1)
                ratio_match = np.transpose(np.array([range(reg_ratio.shape[0]), range(reg_ratio.shape[0])]),
                                           axes=[1, 0])
            else:
                ratio_match = PointMatch.ratio_match(
                    reg_ratio[:, 0].tolist(), single_ratios, method='reg')
                #   update matched ratio
                reg_ratio[ratio_match[:, 0], 0] = (reg_ratio[ratio_match[:, 0], 0] +
                                                   update_weight * single_ratios[ratio_match[:, 1]]) /\
                                                  (1 + update_weight)
                #   Add counting
                reg_ratio[ratio_match[:, 0], 1] += 1

            if reg_kp is None:
                reg_kp = [None] * reg_ratio.shape[0]
            if reg_feat is None:
                reg_feat = [None] * reg_ratio.shape[0]
            for ridx, aidx in zip(ratio_match[:, 0].tolist(), ratio_match[:, 1].tolist()):
                #   Register key-points
                if reg_kp[ridx] is None:
                    #   We would add count to the registered key-points
                    #   And give the key-points with highest recall to the database.
                    reg_kp[ridx] = np.concatenate([np.array(single_kps[aidx]),
                                                   np.ones((np.array(single_kps[aidx]).shape[0], 1))], -1)
                    reg_feat[ridx] = np.concatenate([np.array(single_feats[aidx]),
                                                     np.ones((np.array(single_feats[aidx]).shape[0], 1))], -1)
                else:
                    #   match all key-points with
                    kp_match = PointMatch.keypoints_match(reg_kp[ridx][:, :2], np.array(single_kps)[aidx])
                    feat_match = np.array(single_feats)[aidx][kp_match[:, 1]]
                    for i in range(kp_match.shape[0]):
                        #   give all matched key-points a average
                        reg_kp[ridx][kp_match[i, 0], :2] = \
                            (reg_kp[ridx][kp_match[i, 0], :2] +
                             update_weight * np.array(single_kps)[aidx][kp_match[i, 1]]) /\
                            (1 + update_weight)
                        reg_kp[ridx][kp_match[i, 0], -1] += 1
                        #   give all matched feature an OR operation
                        reg_feat[ridx][kp_match[i, 0], :-1] = np.logical_or(reg_feat[ridx][kp_match[i, 0], :-1], feat_match[i])
                        reg_feat[ridx][kp_match[i, 0], -1] += 1


        if self.debug:
            clct_idx = 0
            for img in imglist:
                simg, bimg = self.preproc(img)
                #   Get the Bounding Boxes
                bboxes = self.detector.run(bimg, nms_thresh=0.1)
                #   Cropping & Padding (build batch)
                records = preprocessing.crop_img(bimg, bboxes)
                #   Transpose the list into arg dimension
                _, cimgs = list(map(list, zip(*records)))

                ratios = []
                for cidx in range(len(cimgs)):
                    _kp, _ = self.extractor.run(cimgs[cidx])
                    cimgs[cidx] = self.extractor.visualize(
                        cimgs[cidx], _kp, color=(255, 0, 0))
                    wh = cord_convert.tlbr2cwh(bboxes[cidx])[2:]
                    ratios.append(wh[1] / float(wh[0]))

                ratio_match = PointMatch.ratio_match(
                    reg_ratio[:, 0].tolist(), ratios, method='reg')
                for m in ratio_match:
                    cimgs[m[1]] = self.extractor.visualize(
                        cimgs[m[1]],
                        cord_convert.denorm_point(
                            reg_kp[m[0]][:, :2], cimgs[m[1]].shape[:2]).astype(np.int32),
                        color=(0, 0, 255))

                bimg = preprocessing.decrop_img(bimg, bboxes, cimgs)
                cv2.imwrite('reg'+str(clct_idx+1)+'.jpg',
                            self.detector.visualize(bimg, bboxes))
                clct_idx += 1
        return (reg_ratio, reg_kp, reg_feat), True, 'Success'

    def authenticate(self, img, model, min_poi=6, metric='logic'):
        """
        Authentication process
        match the feature extracted from input image with the info in model.
                     `ratio_match`                    `kp_match`
        Ratio match ===============> Key-point match ============> feature match

        *   count the matched key-point and calculate the mean one
        *   give the confident ratio estimation by grab top-k result

        :param imglist:         list of images
        :param min_poi:         minimum PoI number
        :return:
        """
        reg_ratio, reg_kp, reg_feat = model
        assert reg_feat is not None and reg_kp is not None and reg_feat is not None
        _, _, ratios, kps, feats = self.extract(img)

        #   To simplify the process, we only accept a set of pictures
        #   We can do more than this, online-registration process actually is available
        single_ratios = np.array(ratios)
        #   match ratios
        ratio_match = PointMatch.ratio_match(
            reg_ratio[:, 0].tolist(), single_ratios, method='reg')

        matched = 0
        for ridx, aidx in zip(ratio_match[:, 0].tolist(), ratio_match[:, 1].tolist()):
            #   Register key-points
            #   match all key-points with
            kp_match = PointMatch.keypoints_match(
                reg_kp[ridx][:, :2], np.array(kps)[aidx])
            #   feature match criterion
            if kp_match.shape[0] != 0 and kp_match.shape[1] != 0:
                if metric == 'logic':
                    #   Logical match
                    m = np.mean(np.logical_not(np.logical_xor(reg_feat[ridx][kp_match[:, 0], :-1], feats[aidx][kp_match[:, 1]])))
                    if m > 0.7:
                        matched += 1
                elif metric in ['l1', 'l2']:
                    ord = int(metric.split('l')[-1])
                    diff = reg_feat[ridx][kp_match[:, 0], :-1] - feats[aidx][kp_match[:, 1]]
                    diff_norm = np.array(list(map(lambda x: np.linalg.norm(x, ord=ord), list(diff))))
                    m = np.mean(diff_norm)
                    if metric == 'l1':
                        m_threshold = 10
                    elif metric == 'l2':
                        m_threshold = 5
                    if m < m_threshold and m != 0:
                        matched += 1
        if self.debug:
            clct_idx = 0
            simg, bimg = self.preproc(img)
            #   Get the Bounding Boxes
            bboxes = self.detector.run(bimg, nms_thresh=0.1)
            #   Cropping & Padding (build batch)
            records = preprocessing.crop_img(bimg, bboxes)
            #   Transpose the list into arg dimension
            _, cimgs = list(map(list, zip(*records)))

            ratios = []
            for cidx in range(len(cimgs)):
                _kp, _ = self.extractor.run(cimgs[cidx])
                cimgs[cidx] = self.extractor.visualize(
                    cimgs[cidx], _kp, color=(255, 0, 0))
                wh = cord_convert.tlbr2cwh(bboxes[cidx])[2:]
                ratios.append(wh[1] / float(wh[0]))

            ratio_match = PointMatch.ratio_match(
                reg_ratio[:, 0].tolist(), ratios, method='reg')
            for m in ratio_match:
                cimgs[m[1]] = self.extractor.visualize(
                    cimgs[m[1]],
                    cord_convert.denorm_point(
                        reg_kp[m[0]][:, :2], cimgs[m[1]].shape[:2]).astype(np.int32),
                    color=(0, 0, 255))

            bimg = preprocessing.decrop_img(bimg, bboxes, cimgs)
            cv2.imwrite('reg' + str(clct_idx + 1) + '.jpg',
                        self.detector.visualize(bimg, bboxes))

        # validate the matched ratios and features and give the result
        ret = True
        if matched < min_poi:
            return False, 'Mismatched'
        else:
            return True, 'Success'


if __name__ == '__main__':
    d = ContourBox.ContourBox()

    e = HarrisLBP.HarrisLBP()
    test = CrossClassTest.CrossClassTest(base_dir='samples/digit_data')

    client = HandWritingAuthInstance(d, e, debug=True)
    reg_info, status, status_info = client.register(test.classes[0], min_poi=6, update_weight=0.8)

    #
    serialized = TranslateLayer.TranslateLayer().serialize(reg_info)
    deserialized = TranslateLayer.TranslateLayer().deserialize(serialized)
    #print(deserialized)
    #print((deserialized[0] == reg_info[0]).all())
    print((deserialized[1][0] == reg_info[1][0]).all())
    ret = client.authenticate(test.classes[0][0], deserialized, min_poi=8)
    print(ret)
    ret = client.authenticate(test.classes[1][0], deserialized, min_poi=8)
    print(ret)
    ret = client.authenticate(test.classes[2][0], deserialized, min_poi=8)
    print(ret)
