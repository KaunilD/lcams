import cv2
import sys
import numpy as np
#import matplotlib.pyplot as plt
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
    elif len(faces) == 2:
        rect1 = faces[0]
        rect2 = faces[1]
        return list([True, img[rect1[0]:rect1[0]+rect1[2], rect1[1]:rect1[1]+rect1[3]], img[rect2[0]:rect2[0]+rect2[2], rect2[1]:rect2[1]+rect2[3]] ])

def loadClassifier():
    # Load Haar cascade files containing features
    #
    cascPaths = ['/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/haarcascade_frontalface_alt.xml']
    faceCascades = []
    for casc in cascPaths:
        faceCascades.append(cv2.CascadeClassifier(casc))
    return faceCascades[0]

def main():

    train_img_combined = list()
    train_lable_combined = list()
    test_img_combined = list()
    test_lable_combined = list()
    faceCascade = loadClassifier()

    dataset = np.load("/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/dataset/dataset_kde.npz")
    train_img, train_lable = dataset["train_img"], dataset["train_lable"]
    test_img, test_lable = dataset["test_img"], dataset["test_lable"]
    for i in range(0, len(train_img)):
        img = train_img[i]
        img = cv2.equalizeHist(img)

        faces = pre_process(img, faceCascade)
        faces[1] = faces[1].astype(np.float32)
        faces[1] /= np.max(faces[1])
        faces[1] = np.asarray(faces[1])
        faces[1] = cv2.resize(faces[1], (128, 128))

        train_img_combined.append([faces[1]])
        train_lable_combined.append(train_lable[i])

    # jaffe
    dataset = np.load("/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/dataset/dataset_jaffe.npz")
    train_img, train_lable = dataset["train_img"], dataset["train_lable"]
    test_img, test_lable = dataset["test_img"], dataset["test_lable"]
    for i in range(0, len(train_img)):
        img = train_img[i]
        img = cv2.equalizeHist(img)
        faces = pre_process(img, faceCascade)

        faces[1] = faces[1].astype(np.float32)
        faces[1] /= np.max(faces[1])
        faces[1] = np.asarray(faces[1])
        faces[1] = cv2.resize(faces[1], (128, 128))

        train_img_combined.append([faces[1]])
        train_lable_combined.append(train_lable[i])
    # ck+
    dataset = np.load("/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/dataset/dataset_ck+.npz")
    train_img, train_lable = dataset["train_img"], dataset["train_lable"]
    test_img, test_lable = dataset["test_img"], dataset["test_lable"]
    for i in range(0, len(train_img)):
        img = train_img[i]
        img = cv2.equalizeHist(img)

        faces = pre_process(img, faceCascade)

        faces[1] = faces[1].astype(np.float32)
        faces[1] /= np.max(faces[1])
        faces[1] = np.asarray(faces[1])
        faces[1] = cv2.resize(faces[1], (128, 128))
        faces[1] = np.array([faces[1]])

        train_img_combined.append(faces[1])
        train_lable_combined.append(train_lable[i])
    np.savez("/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/dataset/preprocess_ds", train_img = train_img_combined, test_img = test_img_combined, train_lable = np.asarray(train_lable_combined, dtype=np.int32), test_lable = test_lable_combined)

if __name__=="__main__":
    main()
