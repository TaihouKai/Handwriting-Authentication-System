#coding: utf-8

#   Generate empty config

import sys
sys.path.append('..')
from net.AlexNet import AlexNetConfig

config = AlexNetConfig

if __name__ == '__main__':
    c = config()
    c.create_empty('../config/empty.cfg')
