import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from train.models.MLP import MLP
from train.models.CustomNet import CustomNet
import cv2
import argparse
import math
import matplotlib.pyplot as plt
import glob
import shutil
#python predict.py --params params/epoch-125.model --image

EMOTIONS = ["neutral",  "happiness", "anger", "sadness"]

def pre_process(img, cascade):
    img = cv2.resize(img, (0,0), fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
    faces = cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    if len(faces) == 0:
        return list([False])
    elif len(faces) == 1:
        rect = faces[0]
        return list([True, img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]])
    elif len(faces) >= 2:
        rect1 = faces[0]
        rect2 = faces[1]
        return list([True, img[rect1[0]:rect1[0]+rect1[2], rect1[1]:rect1[1]+rect1[3]], img[rect2[0]:rect2[0]+rect2[2], rect2[1]:rect2[1]+rect2[3]] ])

def loadClassifier():
    # Load Haar cascade files containing features
    #/home/kaunildhruv/fbsource/fbcode/experimental/themachinist/ml/autoencoders/
    cascPaths = ['data/haarcascade_frontalface_alt.xml']
    faceCascades = []
    for casc in cascPaths:
        faceCascades.append(cv2.CascadeClassifier(casc))
    return faceCascades[0]




class Classifier(chainer.Chain):
    def __init__(self, model):
        super(Classifier, self).__init__(predictor=model)
    def __call__(self, x, t):
        return F.softmax(self.predictor(x, t))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--crop_face', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--folder', type=str)

    args = parser.parse_args()
    print("Loading model..")
    if args.model == 'mlp':
        model = MLP()
    elif args.model == 'custom':
        model = CustomNet()
    model.train = False
    model = Classifier(model)
    model.train = False
    serializers.load_npz(args.params, model )
    print("Model loaded..")
    faceCascade = loadClassifier()
    count = 0
    if args.folder:
        images = glob.glob(args.folder+'/*.png')

        for image in images:
            img = cv2.imread(image, 0)
            img = pre_process(img, faceCascade)
            img = img[1].astype(np.float32)
            img /= np.max(img)
            img = cv2.resize(img, (128, 128))
            plt.imsave("face_norm.png", img)

            img = np.array([[img]])

            img_chain = Variable(np.array(img, dtype=np.float32))
            pred = model(img_chain, None)
            print(pred.data[0], EMOTIONS[np.argmax(pred.data[0])])
            if np.argmax(pred.data[0]) == 2:
                count+=1
            print(count)
    else:
        if args.crop_face == 1:
            img = pre_process(img, faceCascade)
            img = img[1].astype(np.float32)
        else:
            print("Not cropping face")
            img = img.astype(np.float32)
        img /= np.max(img)
        img = cv2.resize(img, (128, 128))
        plt.imsave("face_norm.png", img)

        img = np.array([[img]])

        img_chain = Variable(np.array(img, dtype=np.float32))
        #pred = model(img_chain, None)
        pred = model(img_chain, None)
        print(pred.data[0], EMOTIONS[np.argmax(pred.data[0])])
        #plt.imsave("pred_l1.png", img)
