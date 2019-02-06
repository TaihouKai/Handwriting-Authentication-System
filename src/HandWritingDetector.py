#coding: utf-8
import os 
import sys
import cv2
import tensorflow as tf

sys.path.append('..')
from net.AlexNet import AlexNet, AlexNetConfig

class HandwritingDetector():
    """
    Handwriting detection
    -   Attributes:
        -   Net_in_size

    """
    def __init__(self, edge_model):
        """
        
        """
        self.edge_gen = cv2.ximgproc.createStructuredEdgeDetection(model = edge_model)
        self.bbox_gen = cv2.ximgproc.createEdgeBoxes(maxBoxes = 1000,
                                                    alpha = 0.65,
                                                    beta = 0.75,
                                                    minScore = 0.03)
    
    def generate_edge(self, img):
        edgearray = self.edge_gen.detectEdges(img)
        orientationarray = self.edge_gen.computeOrientation(edgearray)
        suppressed_edgearray = self.edge_gen.edgesNms(edgearray, orientationarray)
