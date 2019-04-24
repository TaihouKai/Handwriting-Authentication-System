#coding: utf-8

#   Generate empty config

"""
Fangrui Liu     mpskex@github   mpskex@163.com
Department of Computer Science and Technology
Faculty of Information
Beijing University of Technology
Copyright 2019
"""

import sys
sys.path.append('..')
from classifiers.AlexNet import AlexNetConfig

config = AlexNetConfig

if __name__ == '__main__':
    c = config()
    c.create_empty('../config/empty.cfg')
