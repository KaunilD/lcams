import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from train.models import MLP
from train.models import CustomNet
import cv2
import argparse
import math
import logging
import json

EMOTIONS = ["neutral",  "happiness", "anger", "sadness"]

history = [0, 0, 0, 0]

class Classifier(chainer.Chain):
    def __init__(self, model):
        super(Classifier, self).__init__(predictor=model)
    def __call__(self, x, t):
        return F.softmax(self.predictor(x, t))

def pre_process(img, cascade):
    img = cv2.resize(img, (0,0), fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    #logging.warn("Found {0} faces!".format(len(faces)))
    if len(faces) == 0:
        return list([False])
    elif len(faces) == 1:
        rect = faces[0]
        return list([True, img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]])
    elif len(faces) >= 2:
        rect1 = faces[0]
        rect2 = faces[1]
        return list([True, img[rect1[0]:rect1[0]+rect1[2], rect1[1]:rect1[1]+rect1[3]], img[rect2[0]:rect2[0]+rect2[2], rect2[1]:rect2[1]+rect2[3]] ])


def loadCascades():
    # Load Haar cascade files containing features
    logging.info('Loading haar-cascades.')
    cascPaths = ['train/datasetpreparation/haarcascades/haarcascade_frontalface_alt.xml']
    faceCascades = []
    for casc in cascPaths:
        faceCascades.append(cv2.CascadeClassifier(casc))
    return faceCascades[0]

def createArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str)
    parser.add_argument('--video', type=str)
    return parser.parse_args()

def loadModel(model_params):
    logging.info("Loading model.")
    model = MLP.MLP()
    model.train = False
    model = Classifier(model)
    model.train = False
    serializers.load_npz(model_params, model )
    logging.info("Model loaded.")
    return model

def prepareLogger():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)


def classify_image(img):

    faceimage = img[1].astype(np.float32)
    faceimage /= np.max(faceimage)
    faceimage = cv2.resize(faceimage, (128, 128))
    faceimage = np.array([[faceimage]])
    faceimage_var = Variable(np.array(faceimage, dtype=np.float32))
    pred = model(faceimage_var, None)

    logging.info( EMOTIONS[ np.argmax(pred.data[0]) ]  )

    history[np.argmax(pred.data[0])]+=1


def main():
    args = createArgs()
    prepareLogger()
    global model
    model = loadModel(args.params)

    faceCascade = loadCascades()
    cv2.namedWindow('Video')
    cap_vid = cv2.VideoCapture(args.video)
    cap_face = cv2.VideoCapture(0)

    if cap_vid.isOpened():
        rval_vid, frame_vid  = cap_vid.read()
    else:
        rval_vid = False

    f_no = 0
    while(rval_vid):
        f_no+=1
        rval_vid, frame_vid = cap_vid.read()
        key = cv2.waitKey(1)
        # ESC for exit
        if key == 27 or f_no > 40:
            break
        if (f_no%10 == 0):

            rval_face, frame_face = cap_face.read()
            frame = pre_process(frame_face, faceCascade)
            if frame[0]:
                classify_image(frame)
        cv2.imshow('Video',frame_vid)
    cap_face.release()
    cap_vid.release()

    avg_emotion = np.argmax(history) + 1
    movie_db = json.load(open('data/db.json'))
    logging.info('Overall user was {}, we recommend him : {}'.format(EMOTIONS[avg_emotion-1], movie_db[str(avg_emotion)]))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
