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
import subprocess, csv
from constants import *
from PIL import Image, ImageDraw, ImageFont
global face, faces
import time

recognizer_gen = cv2.face.LBPHFaceRecognizer_create()

recognizer_gen.read("genero.yml")
print("creado_genero")
subjects_gen = ["femenino" , "masculino"]

recognizer_edad = cv2.face.LBPHFaceRecognizer_create()

recognizer_edad.read("edad.yml")
print("creado_edad")
subjects_edad = ["adulto" , "joven", "viejo","nino" ]
ultim_faces=None
fr=0
fem = porc_fem = mas =porc_masc= adulto=joven=viejo=nino=p_adulto=p_joven=p_nino=p_viejo=0
neutral =feliz =triste =p_neutral=p_feliz=p_triste=0
#################

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#font = cv2.FONT_HERSHEY_SIMPLEX


video_capture = cv2.VideoCapture(0)
feelings_faces = []


#####################
def porc_genero(genero):
	global fem
	global mas
	global porc_fem
	global porc_masc
	if(genero=="femenino"):
		fem+=1
	elif(genero=="masculino"):
		mas+=1
	porc_fem=int(float(fem)/(fem+mas)*100)
	porc_masc=100-porc_fem


def porc_edad(edad):
	global adulto
	global joven
	global viejo
	global nino
	global p_adulto
	global p_joven
	global p_nino
	global p_viejo
	if (edad=="adulto"):
		adulto+=1

	elif (edad=="joven"):
		joven+=1
	elif (edad=="viejo"):
		viejo+=1
		
	elif (edad=="nino"):
		nino+=1

	p_adulto=int(float(adulto)/(adulto+joven+viejo+nino)*100)
	p_joven=int(float(joven)/(adulto+joven+viejo+nino)*100)
	p_viejo=int(float(viejo)/(adulto+joven+viejo+nino)*100)
	p_nino=int(float(nino)/(adulto+joven+viejo+nino)*100)
	

def porc_emo(emocion):
	global neutral
	global feliz
	global triste
	global p_neutral
	global p_feliz
	global p_triste

	if (emocion=="Feliz"):
		feliz+=1

	elif (emocion=="Neutral"):
		neutral+=1
	elif (emocion=="Triste"):
		triste+=1

	p_feliz=int(float(feliz)/(feliz+neutral+triste)*100)
	p_neutral=int(float(neutral)/(feliz+neutral+triste)*100)
	p_triste=int(float(triste)/(feliz+neutral+triste)*100)

	



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
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5);
    
    #if no faces are detected then return original img
	if (len(faces) == 0):
		return None
    #num_face=0
    #for (x,y,w,h) in faces:
        #extract the face area
     #   (x, y, w, h) = faces[num_face]
      #  num_face+=
    #return only the face part of the image
    #return gray[y:y+w, x:x+h], faces[num_face]
	return faces


def predict(test_img,hora):
	global ultim_faces
	global fr
    #make a copy of the image as we don't want to chang original image
	img = test_img
	gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect face from the image
	faces  = detect_face(img)
	#print(faces)
	#print("ultima: ")
	#print(ultim_faces)
	if(fr>=5):
		fr=0
		if (faces is not None and ultim_faces is not None):
			for (x,y,w,h) in faces:
				for (a,b,c,d) in ultim_faces:
					if (y<(b+0.2*b) and y>(b-0.2*b) and x<(a+0.2*a) and x>(a-0.2*a)):
						print("misma cara")
						rect=(x,y,w,h)
						label_gen= recognizer_gen.predict(gray2[y:y+h,x:x+w])
						label_edad= recognizer_edad.predict(gray2[y:y+h,x:x+w])
						#print(label)
							#get name of respective label returned by face recognizer
						label_text_gen = subjects_gen[label_gen[0]]
						label_text_edad = subjects_edad[label_edad[0]]
						result = network.predict(format_image(img))
								 
						text_emo = EMOTIONS[np.argmax(result)]
								 #draw a rectangle around face detected
						draw_rectangle(img, rect)
							#draw name of predicted person
						draw_text(img, label_text_gen+" "+label_text_edad+" "+text_emo, rect[0], rect[1]-5)
					else:
						
						rect=(x,y,w,h)
						label_gen= recognizer_gen.predict(gray2[y:y+h,x:x+w])
						label_edad= recognizer_edad.predict(gray2[y:y+h,x:x+w])
						#print(label)
							#get name of respective label returned by face recognizer
						label_text_gen = subjects_gen[label_gen[0]]
						label_text_edad = subjects_edad[label_edad[0]]
						result = network.predict(format_image(img))
								 
						text_emo = EMOTIONS[np.argmax(result)]
								 #draw a rectangle around face detected
						draw_rectangle(img, rect)
							#draw name of predicted person
						draw_text(img, label_text_gen+" "+label_text_edad+" "+text_emo, rect[0], rect[1]-5)
						porc_genero(label_text_gen)
						porc_edad(label_text_edad)
						porc_emo(text_emo)
						csv_writer.writerow([label_text_gen,label_text_edad,text_emo, str(hora)])
						print("segunda")
						csvfile.flush()
			ultim_faces = faces
		elif(ultim_faces is None and faces is not None):
			for (x,y,w,h) in faces:
					rect=(x,y,w,h)
					label_gen= recognizer_gen.predict(gray2[y:y+h,x:x+w])
					label_edad= recognizer_edad.predict(gray2[y:y+h,x:x+w])
					#print(label)
						#get name of respective label returned by face recognizer
					label_text_gen = subjects_gen[label_gen[0]]
					label_text_edad = subjects_edad[label_edad[0]]
					result = network.predict(format_image(img))
							 
					text_emo = EMOTIONS[np.argmax(result)]
							 #draw a rectangle around face detected
					draw_rectangle(img, rect)
						#draw name of predicted person
					draw_text(img, label_text_gen+" "+label_text_edad+" "+text_emo, rect[0], rect[1]-5)
					porc_genero(label_text_gen)
					porc_edad(label_text_edad)
					print(text_emo)
					porc_emo(text_emo)
					csv_writer.writerow([label_text_gen,label_text_edad,text_emo, str(hora)])
					csvfile.flush()
			ultim_faces = faces
			
		elif(faces is None):
			print("no se encontraron rostros")
			ultim_faces = faces

	if (faces is not None and ultim_faces is not None):
		if (len(faces)!= len(ultim_faces)):
			fr+=1
			for (x,y,w,h) in faces:
				rect=(x,y,w,h)
				label_gen= recognizer_gen.predict(gray2[y:y+h,x:x+w])
				label_edad= recognizer_edad.predict(gray2[y:y+h,x:x+w])
				#print(label)
				#get name of respective label returned by face recognizer
				label_text_gen = subjects_gen[label_gen[0]]
				label_text_edad = subjects_edad[label_edad[0]]
				result = network.predict(format_image(img))
								 
				text_emo = EMOTIONS[np.argmax(result)]
							 #draw a rectangle around face detected
				draw_rectangle(img, rect)
							#draw name of predicted person
				draw_text(img, label_text_gen+" "+label_text_edad+" "+text_emo, rect[0], rect[1]-5)
			ultim_faces = faces


		if (len(faces)== len(ultim_faces)):
			fr=0
			for (x,y,w,h) in faces:
						rect=(x,y,w,h)
						label_gen= recognizer_gen.predict(gray2[y:y+h,x:x+w])
						label_edad= recognizer_edad.predict(gray2[y:y+h,x:x+w])
						#print(label)
							#get name of respective label returned by face recognizer
						label_text_gen = subjects_gen[label_gen[0]]
						label_text_edad = subjects_edad[label_edad[0]]
						result = network.predict(format_image(img))
								 
						text_emo = EMOTIONS[np.argmax(result)]
								 #draw a rectangle around face detected
						draw_rectangle(img, rect)
							#draw name of predicted person
						draw_text(img, label_text_gen+" "+label_text_edad+" "+text_emo, rect[0], rect[1]-5)
						
			ultim_faces = faces
	
	elif(ultim_faces is None):
		if(faces is None):
			fr+=1
			ultim_faces=None
		
		
		elif (faces is not None):
			fr+=1
			
			for (x,y,w,h) in faces:
						rect=(x,y,w,h)
						label_gen= recognizer_gen.predict(gray2[y:y+h,x:x+w])
						label_edad= recognizer_edad.predict(gray2[y:y+h,x:x+w])
						#print(label)
							#get name of respective label returned by face recognizer
						label_text_gen = subjects_gen[label_gen[0]]
						label_text_edad = subjects_edad[label_edad[0]]
						result = network.predict(format_image(img))
								 
						text_emo = EMOTIONS[np.argmax(result)]
								 #draw a rectangle around face detected
						draw_rectangle(img, rect)
							#draw name of predicted person
						draw_text(img, label_text_gen+" "+label_text_edad+" "+text_emo, rect[0], rect[1]-5)
						
			#ultim_faces = faces
	elif(faces is None):
		fr+=1
		ultim_faces=None
		

	return img

############################################
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

####################################

def format_image(image):
    global face, faces
        
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)



    faces = face_cascade.detectMultiScale(
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

#############################################
network = EmotionRecognition()
network.build_network()
a=0
ultima_faces= None
hora = subprocess.check_output(["date", "+%H-%M-%S"])  
#namedWindow("img" , CV_WINDOW_AUTOSIZE)
with open('csv_files/'+str(hora)+'.csv', 'w') as csvfile:
	csv_writer = csv.writer(csvfile)
	while (a<80 ):

		hora = subprocess.check_output(["date", "+%H-%M-%S"]) 
		for index, emotion in enumerate(EMOTIONS):

		        feelings_faces.append(cv2.imread(emotion, -1))


		
		ret, test_img1 = video_capture.read()

	   
	   
	   
		print("Predicting images...")




	#perform a prediction
		predicted_img1 = predict(test_img1,hora)
		#ultima_faces=faces
		print("masculino "+str(porc_masc))
		print("joven  "+str(p_joven))
		print("neutral "+str(p_neutral))
		  
		#cv2.imshow('img',predicted_img1 )


		a+=1

		

		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


#miguel perez migaperezber@hotmail.com
