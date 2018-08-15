#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os, sys
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np
import re
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
import random
from constants import *
from PIL import Image, ImageDraw, ImageFont
global face, faces
import time

recognizer_gen = cv2.face.LBPHFaceRecognizer_create()
print("creado_genero")
recognizer_gen.read("genero.yml")

subjects_gen = ["femenino" , "masculino"]

recognizer_edad = cv2.face.LBPHFaceRecognizer_create()
print("creado_edad")
recognizer_edad.read("edad.yml")

subjects_edad = ["adulto" , "joven", "viejo","nino" ]


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)


def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    #num_face=0
    #for (x,y,w,h) in faces:
        #extract the face area
     #   (x, y, w, h) = faces[num_face]
      #  num_face+=
    #return only the face part of the image
    #return gray[y:y+w, x:x+h], faces[num_face]
    return faces
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect face from the image
    faces  = detect_face(img)
    
    for (x,y,w,h) in faces:
    #predict the image using our face recognizer 
        rect=(x,y,w,h)
        label_gen= recognizer_gen.predict(gray2[y:y+h,x:x+w])
        label_edad= recognizer_edad.predict(gray2[y:y+h,x:x+w])
        #print(label)
        #get name of respective label returned by face recognizer
        label_text_gen = subjects_gen[label_gen[0]]
        label_text_edad = subjects_edad[label_edad[0]]
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text_gen+" "+label_text_edad, rect[0], rect[1]-5)
        
    return img

print("Predicting images...")

#load test images
test_img1 =cv2.imread('familia.jpg',1)


#perform a prediction
predicted_img1 = predict(test_img1)
print("imagen lista")

cv2.imshow('img',predicted_img1 )

cv2.waitKey(0)      
    
cap.release()

cv2-destroyAllWindows()
