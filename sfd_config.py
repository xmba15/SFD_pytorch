#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import torch


_DIRECTORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))
_MODEL_PATH = os.path.join(_DIRECTORY_ROOT, "models")
_MODEL = os.path.join(_MODEL_PATH, "s3fd_convert.pth")
_DATA_PATH = os.path.join(_DIRECTORY_ROOT, "data")
_TEST_IMAGE = os.path.join(_DATA_PATH, "test01.jpg")


class SFDConfig(object):
    def __init__(self):
        self.MODEL = _MODEL
        self.USE_CUDA = torch.cuda.is_available()
        self.TEST_IMAGE = _TEST_IMAGE

    def display(self):
        """
        Display Configuration values.

        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
