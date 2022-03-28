import json
import cv2
from glob import glob
from os import path

from util import *
from extractor import *
from detector import *
from ClientModel import HandWritingAuthInstance

imgs = [cv2.imread(f)[:,:,:3] for f in glob(path.join('outimages', 'YueYu-*.png'))]
print(len(imgs))

if __name__ == '__main__':
    # 检测 BBox 的一些参数：你也可以随意调整一下 (bin_threshold=0.7, kernel_size=7, blur=cv2.GaussianBlur)
    d = ContourBox.ContourBox()
    # 提取特征的一些参数: 你可以随意调整一下 (ksize=3, block_size=3, k=0.16, blur=cv2.GaussianBlur)
    e = HarrisLBP.HarrisLBP()
    i = HandWritingAuthInstance(d, e, debug=True)

    # 注册的一些参数
    # 返回: 注册信息，注册状态，注册状态信息
    reg_info, status, status_info = i.register([imgs[0]], min_poi=6, update_weight=0.0)

    ret = i.authenticate(imgs[0], reg_info, min_poi=8)
    print(ret)