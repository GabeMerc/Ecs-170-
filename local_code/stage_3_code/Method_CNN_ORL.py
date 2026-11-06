'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    data = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        #TODO: build conv layers

    def forward(self, x):
        #TODO
        pass

    def train(self, X, y):
        #TODO
        pass

    def test(self, X):
        #TODO
        pass

    def run(self):
        #TODO
        pass
