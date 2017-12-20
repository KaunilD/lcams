import numpy as np
import csv
import shutil, os, glob
import cv2
#'neutral', 'happy', 'anger', 'sadness'
def main():
    #kde

    train_img = list()
    test_img = list()

    train_lables = list()
    test_lables = list()
    # neutral
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/neutral/*.JPG'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(0)

            test_img.append(img)
            test_lables.append(0)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)

    # happy
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/happy/*.JPG'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(1)

            test_img.append(img)
            test_lables.append(1)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)

    # anger
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/anger/*.JPG'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(2)

            test_img.append(img)
            test_lables.append(2)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)
    # sad
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/sad/*.JPG'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(3)

            test_img.append(img)
            test_lables.append(3)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)
    np.savez("/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/dataset/dataset_kde", train_img = train_img, test_img = test_img, train_lable = train_lables, test_lable = test_lables)



    #jaffe
    train_img = list()
    test_img = list()

    train_lables = list()
    test_lables = list()
    # neutral
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/neutral/*.tiff'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(0)

            test_img.append(img)
            test_lables.append(0)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)

    # happy
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/happy/*.tiff'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(1)

            test_img.append(img)
            test_lables.append(1)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)

    # anger
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/anger/*.tiff'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(2)

            test_img.append(img)
            test_lables.append(2)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)
    # sad
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/sad/*.tiff'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(3)

            test_img.append(img)
            test_lables.append(3)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)
    np.savez("/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/dataset/dataset_jaffe", train_img = train_img, test_img = test_img, train_lable = train_lables, test_lable = test_lables)

    #ck+


    train_img = list()
    test_img = list()

    train_lables = list()
    test_lables = list()
    # neutral
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/neutral/*.png'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(0)

            test_img.append(img)
            test_lables.append(0)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)

    # happy
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/happy/*.png'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(1)

            test_img.append(img)
            test_lables.append(1)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)

    # anger
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/anger/*.png'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(2)

            test_img.append(img)
            test_lables.append(2)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)
    # sad
    dataset_path = '/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/datasetpreparation/data/facial_expressions_filtered/sad/*.png'
    image_files = glob.glob(dataset_path)
    print(image_files)
    print()
    for image in image_files:
        try:
            #load image in grayscale
            img = cv2.imread(image, 0)
            img = cv2.equalizeHist(img)
            train_img.append(img)
            train_lables.append(3)

            test_img.append(img)
            test_lables.append(3)
            #test.append([np.array(img), b])
        except ValueError as vex:
            print(vex)
    np.savez("/Users/kaunildhruv/Desktop/themachinist/ml/facial_emotion_recognition/dataset/dataset_ck+", train_img = train_img, test_img = test_img, train_lable = train_lables, test_lable = test_lables)


if __name__=="__main__":
    main()
