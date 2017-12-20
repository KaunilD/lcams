import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F
import cv2
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
           l1 = L.Linear(16384, 4096),
           l2 = L.Linear(4096, 1024),
           l3 = L.Linear(1024, 256),
           l4 = L.Linear(256, 64),
           l5 = L.Linear(64, 16),
           l7 = L.Linear(16, 4)
        )
        self.train = True
    def __call__(self, x, t):
       # Forward pass
       h = self.l1(x)
       h = self.l2(h)
       h = self.l3(h)
       h = self.l4(h)
       h = self.l5(h)
       y = self.l7(h)
       if self.train:
            self.loss = F.softmax_cross_entropy(y, t)
            self.acc = F.accuracy(y, t)
            return self.loss
       else:
            self.pred = F.softmax(y)
            return self.pred
