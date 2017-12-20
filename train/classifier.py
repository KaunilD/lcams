import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class Classifier(Chain):
     def __init__(self, predictor):
         super(Classifier, self).__init__(predictor=predictor)
         self.train = True
     def __call__(self, x, t):
         y = self.predictor(x, t)
         report({'loss': y}, self)
         return y
