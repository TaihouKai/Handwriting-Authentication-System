#coding: utf-8

import cv2
import time
import numpy as np
import sys
sys.path.append('..')

import util.nms
import util.cord_convert

class EdgeBox():
    """
    Handwriting detection
    -   Attributes:
        -   Net_in_size

    """
    def __init__(self, edge_model):
        """
        
        """
        self.edge_gen = cv2.ximgproc.createStructuredEdgeDetection(model = edge_model)
        self.bbox_gen = cv2.ximgproc.createEdgeBoxes(maxBoxes = 100,
                                                    alpha = 0.65,
                                                    beta = 0.75,
                                                    minScore = 0.03)
    
    def scale(self, img, factor=0.5):
        return cv2.resize(img, (int(factor*img.shape[1]), int(factor*img.shape[0])))

    def generate_edge(self, img):
        """ Return detected edges, orientations and suppressed edges
        """
        rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edgearray = self.edge_gen.detectEdges(img/255.0)
        orientationarray = self.edge_gen.computeOrientation(edgearray)
        suppressed_edgearray = self.edge_gen.edgesNms(edgearray, orientationarray)
        return edgearray, orientationarray, suppressed_edgearray
    
    def run(self, img):
        """ Return bounding boxes
        """
        _, orientationarray, suppressed_edgearray = self.generate_edge(img)
        return self.bbox_gen.getBoundingBoxes(suppressed_edgearray, orientationarray)
    
    def generate(self, img, factor=0.5):
        img = self.scale(img, factor=factor)
        img = img.astype(np.float32)
        return img, self.run(img)
    
    def visualize(self, img, scale=0.5):
        rimg, bboxes = self.generate(img, factor=scale)
        bboxes = util.cord_convert.cwh2tlbr(bboxes, tolist=False)
        #'''
        bboxes = util.nms.non_max_suppression_fast(bboxes, 0.3)
        #'''
        return __draw_bbox__(rimg, bboxes)


def __draw_bbox__(img, bboxes, color=(255, 0, 0)):
    """ Draw Bounding box
    Args:
        img     :   input image
        bboxes  :   list of bounding boxes for this image
    Return:
        img     :   image with bbox visualization
    """
    for bbox in bboxes:
        bbox = np.array(bbox).astype(np.int16)
        cv2.rectangle(img, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), color)
    return img
    

if __name__ == '__main__':
    e = EdgeBox('../model/RFStructuedEdgeDetector.yml.gz')
    print('Loaded model')
    t = time.time()
    retimg = e.visualize(cv2.imread('../src/digit_data/lfr_a.png')[:,:,:3], scale=0.3)
    print(time.time()-t)
    cv2.imshow('edgebox', retimg.astype(np.uint8))
    cv2.waitKey(0)


