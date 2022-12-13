#-*- coding: utf-8 -*-
# Run in server

import os.path
import sys
PythonPath = os.path.join(os.path.expanduser('~'),'PlasticRings', 'python')
sys.path.append(os.path.abspath(PythonPath))
from PlasticMethods import *
import numpy
import cv2


def cropped(image, h=924): # original is 2448x2448 ==> to make 600x600
    cropped_image = image[h:-h, h:-h]
    return cropped_image

def RawProcessing(mode='grayscale_crop_600_resize_240'):
    picnum = 0
    capture_foldername = os.path.join(os.path.expanduser('~'),'PlasticRings', 'Samples', 'Processed', '{}'.format(mode))
    if not os.path.isdir(capture_foldername):
        os.makedirs(capture_foldername)
    ###
    defective_folder_path = RawPicturesPath(defective=True)
    nondefective_folder_path = RawPicturesPath(defective=False)
    defective_paths = os.listdir(defective_folder_path)
    nondefective_paths = os.listdir(nondefective_folder_path)
    defective_paths = sorted(defective_paths)
    nondefective_paths = sorted(nondefective_paths)
    defective_files = [os.path.join(defective_folder_path, dp) for dp in defective_paths]
    nondefective_files = [os.path.join(nondefective_folder_path, ndp) for ndp in nondefective_paths]
    for ffp in defective_files:
        image = cv2.imread(ffp, cv2.IMREAD_UNCHANGED)
        print('Read picture {}'.format(picnum))
        if mode == 'grayscale_crop_600_resize_240':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cropped(image, h=924)
            image = cv2.resize(image, (0,0), fx=0.4, fy=0.4)
        if mode == 'edges_crop_600_resize_240':
            image = cropped(image, h=924)
            image = cv2.Canny(image,30,55)
            image = cv2.resize(image, (0,0), fx=0.4, fy=0.4)
        capture = "picture_{}_{}_defective_1.jpg".format(str(picnum).zfill(4), mode)
        capture_filename = os.path.join(capture_foldername, capture)
        cv2.imwrite(capture_filename, image)  # save frame as JPEG file
        print('Write: Success') if os.path.exists(capture_filename) else print('Write: Failed')
        picnum += 1
    for ffp in nondefective_files:
        image = cv2.imread(ffp, cv2.IMREAD_UNCHANGED)
        print('Read picture {}'.format(picnum))
        if mode == 'grayscale_crop_600_resize_240':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cropped(image, h=924)
            image = cv2.resize(image, (0,0), fx=0.4, fy=0.4)
        if mode == 'edges_crop_600_resize_240':
            image = cropped(image, h=924)
            image = cv2.Canny(image,30,55)
            image = cv2.resize(image, (0,0), fx=0.4, fy=0.4)
        capture = "picture_{}_{}_defective_0.jpg".format(str(picnum).zfill(4), mode)
        capture_filename = os.path.join(capture_foldername, capture)
        cv2.imwrite(capture_filename, image)  # save frame as JPEG file
        print('Write: Success') if os.path.exists(capture_filename) else print('Write: Failed')
        picnum += 1
    ###

def main():
    RawProcessing(mode='grayscale_crop_600_resize_240')
    RawProcessing(mode='edges_crop_600_resize_240')
    print('Finished!')

if __name__ == '__main__':
    main()