#coding: utf-8

#   Generate empty config

""" 
mpsk's CVDL LayerLibrary @ tf
Version: 2.0

Copyright 2018
"""

import sys
sys.path.append('..')
from classifiers.AlexNet import AlexNetConfig

config = AlexNetConfig

if __name__ == '__main__':
    c = config()
    c.create_empty('../config/empty.cfg')
