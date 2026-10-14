# -*- coding: utf-8 -*-
from typing import Tuple
import cv2 as cv
import numpy as np
from random import randint
def load_models(): 
    # Conf_threshold = 0.5
    # NMS_threshold = 0.5
    backup_path = 'test2'
    net = cv.dnn.readNet(
    backup_path + '/' + 'yolov4-tiny-obj_best.weights',
    backup_path + '/' + 'yolov4-tiny-obj-test.cfg'
    )
    models = cv.dnn_DetectionModel(net)
    models.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    # Пример загрузки моделей из файлов
    # Yolo-модели
    # net = cv2.dnn.readNetFromDarknet('yolo.cfg', 'yolo.weights')
    # yolo_model = cv2.dnn_DetectionModel(net)
    # yolo_model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
    # models = [yolo_model]

    # Пример загрузки модели TensorFlow (не забудьте импортировать библиотеку tensorflow)
    # tf_model = tf.keras.models.load_model('model.h5')
    # models.append(tf_model)
    # models = [yolo_model]
    return models
 
def detect_traffic_light(image, models) -> Tuple:
    Conf_threshold = 0.5
    NMS_threshold = 0.5
    
    with open('test2/obj.names', 'r') as f:
        classes_list = f.read().split('\n')
    
    classes, scores, boxes = models.detect(image, Conf_threshold, NMS_threshold)

    for (classid, score, box) in zip(classes, scores, boxes):
        # cv.rectangle(image, box, (0, 255, 0), 3,) # (box[0], box[1]), (abs(box[2]-box[0]), abs(box[3]-box[1]))
        # i = cv.waitKey(0)
        
        # if classes_list[classid] != "Nothing":
        if classes_list[classid] == "paper":
            continue

        # if True:
        else:
            if score > 0.8:
                cv.rectangle(image, box, (0, 255, 0), 3,)
                cv.putText(image,f"OBJECT:{classes_list[classid]}",(box[0],box[2]),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0, 0, 255),2)
                print(classes_list[classid], score)
        # cv.imshow('s', image)
        # cv.imshow('s', image)
        return classid
    # cv.imshow('s', image)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    detect_traffic_light(frame, load_models())
    cv.imshow('frame', frame)



    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()