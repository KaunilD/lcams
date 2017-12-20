import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F
import cv2
class CustomNet(Chain):
    def __init__(self):
        super(CustomNet, self).__init__(
            conv1=L.Convolution2D(1, 32, 5, stride=1, pad=2),
            conv2=L.Convolution2D(32, 32, 5, stride=1, pad=2),
            conv3=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc4=F.Linear(1344, 4096),
            fc5=F.Linear(4096, 4)
        )
        self.train = True

    def __call__(self, x, t):
       h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
       h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
       h = F.relu(self.conv3(h))
       h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
       h = F.dropout(F.relu(self.fc4(h)), ratio=0.5, train=self.train)
       y = self.fc5(h)
       if self.train:
            self.loss = F.softmax_cross_entropy(y, t)
            self.acc = F.accuracy(y, t)
            return self.loss
       else:
            self.pred = F.softmax(y)
            return self.pred
