#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
from .sfd_config import SFDConfig
from .net_s3fd import s3fd
from .bbox import *


class SFDDetector(object):
    def __init__(self):
        self.config = SFDConfig()
        self.net = s3fd()
        self.init_model(self.config.MODEL)

    def init_model(self, model_path):
        """
        init with pretrained model
        """
        self.net.load_state_dict(torch.load(model_path))
        if self.config.USE_CUDA:
            self.net.cuda()
        self.net.eval()

    def detect(self, img, nms_th=0.3, th=0.5):
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)

        img = Variable(torch.from_numpy(img).float(), volatile=True)
        if self.config.USE_CUDA:
            img = img.cuda()

        BB, CC, HH, WW = img.size()
        olist = self.net(img)

        bboxlist = []
        for i in range(int(len(olist) / 2)):
            olist[i*2] = F.softmax(olist[i*2])
        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(int(len(olist) / 2)):
            ocls, oreg = olist[i*2],olist[i*2+1]
            FB, FC, FH, FW = ocls.size() # feature map size
            stride = 2**(i+2)    # 4,8,16,32,64,128
            anchor = stride*4
            poss = zip(*np.where(ocls[:,1,:,:]>0.05))
            for Iindex, hindex, windex in poss:
                axc,ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0,stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))
        keep = nms(bboxlist, nms_th)
        bboxlist = bboxlist[keep, :]

        return bboxlist

    def test_img(self):
        img = cv2.imread(self.config.TEST_IMAGE)
        bboxlist = self.detect(img)
        for b in bboxlist:
            x1, y1, x2, y2, s = b
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        cv2.imshow('test',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    sfddetector = SFDDetector()
    sfddetector.test_img()


if __name__ == '__main__':
    main()
