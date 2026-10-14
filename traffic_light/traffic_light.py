# -*- coding: utf-8 -*-
from typing import Tuple
import cv2 as cv
import numpy as np
from random import randint


def load_models(): 
    # Conf_threshold = 0.5
    # NMS_threshold = 0.5
    backup_path = 'op'
    net = cv.dnn.readNet(
    backup_path + '/' + 'yolov4-tiny-obj_best.weights',
    backup_path + '/' + 'yolov4-tiny-obj-test.cfg'
    )
    models = cv.dnn_DetectionModel(net)
    models.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    
    return models


def detect_traffic_light(image, models) -> Tuple:
    Conf_threshold = 0.5
    NMS_threshold = 0.5
    
    with open('op/obj.names', 'r') as f:
        classes_list = f.read().split('\n')
    
    classes, scores, boxes = models.detect(image, Conf_threshold, NMS_threshold)

    for (classid, score, box) in zip(classes, scores, boxes):
        cv.rectangle(image, box, (0, 255, 0), 3,) # (box[0], box[1]), (abs(box[2]-box[0]), abs(box[3]-box[1]))
        cv.imshow('s', image)
        i = cv.waitKey(0)
        print(classes_list[classid])
        return classid

image = cv.imread('1114.jpg')
detect_traffic_light(image, load_models())
# if __name__ == "__main__":
#     image = cv.imread('1111.jpg')
#     print(detect_traffic_light(image, load_models()))