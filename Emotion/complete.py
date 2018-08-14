import re
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
import random
import sys
import numpy as np
from constants import *
import cv2
from PIL import Image, ImageDraw, ImageFont
global face, faces
import time
"""
"""
################################################################################################################

class EmotionRecognition:

    def build_network(self):
        # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
        print('[+] Building CNN')
        self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        #self.network = local_response_normalization(self.network)
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(
            self.network, len(EMOTIONS), activation='softmax')
        self.network = regression(
            self.network,
            optimizer='momentum',
            loss='categorical_crossentropy'
        )
        self.model = tflearn.DNN(
            self.network,
            checkpoint_path=SAVE_DIRECTORY + '/emotion_recognition',
            max_checkpoints=1,
            tensorboard_verbose=2
        )
        
        self.load_model()

    
    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        return self.model.predict(image)

    def save_model(self):
        self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
        print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

    def load_model(self):
        if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)):
            self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
            print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)

##########################################################################################

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX
network = EmotionRecognition()
network.build_network()
video_capture = cv2.VideoCapture(0)
feelings_faces = []



def format_image(image):
    global face, faces
        
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)



    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]

    for face in faces:
        

        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
        

    except Exception:
        print("[+] Problem during resize")
        return None
   
    return image





while True:


    for index, emotion in enumerate(EMOTIONS):

            feelings_faces.append(cv2.imread(emotion, -1))


    # Capture frame-by-frame
    
    ret, frame = video_capture.read()

    # Predict result with network
    #result = network.predict(format_image(frame))


    # Write results in frame
    
        

#        face_image = feelings_faces[np.argmax(result[0])]

 #       text = EMOTIONS[np.argmax(result[0])]

  #      cv2.putText(frame, text, (face[0], face[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
                
   #     x, y, w, h = face
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faces = cascade_classifier.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for face in faces:

        (x, y, w, h) = face

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            #print(x, y, w, h)        
        result = network.predict(format_image(frame))
        
        text = EMOTIONS[np.argmax(result)]
        #print(result, text)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

  #      stext = str(text)
#        if stext == 'Feliz':
 #           print( "=D")
 #       except Exception:
  #          print("[+] Problem during resize")
                    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
