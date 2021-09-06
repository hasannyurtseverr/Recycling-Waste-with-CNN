# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:06:12 2020

@author: Hasan YURTSEVER
"""
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Dense, Dropout, Flatten
import cv2 
import glob
import numpy as np
#
def read_img(img_list, img):
    n = cv2.imread(img)
    n=cv2.resize(n,(64,64)) 
    img_list.append(n)
    return img_list
#
#----------train   cam kagit metal plastik
path = glob.glob("cam/*.JPG") #or jpg
list_ = []

cv_image = [read_img(list_, img) for img in path]
path = glob.glob("cam/*.jpeg") #or jpg


cv_image = [read_img(list_, img) for img in path]
path = glob.glob("kagit/*.JPG") #or jpg


cv_image = [read_img(list_, img) for img in path]
path = glob.glob("kagit/*.jpeg") #or jpg


cv_image = [read_img(list_, img) for img in path]
path = glob.glob("metal/*.JPG") #or jpg


cv_image = [read_img(list_, img) for img in path]
path = glob.glob("metal/*.jpeg") #or jpg


cv_image = [read_img(list_, img) for img in path]
path = glob.glob("plastik/*.JPG") #or jpg


cv_image = [read_img(list_, img) for img in path]
path = glob.glob("plastik/*.jpeg") #or jpg


cv_image = [read_img(list_, img) for img in path]
x=np.array(list_)
#---------------
#
#------------test cam kagit metal plastik
path = glob.glob("t_cam/*.JPG") #or jpg
test = []

cv_image = [read_img(test, img) for img in path]
path = glob.glob("t_cam/*.jpeg") #or jpg


cv_image = [read_img(test, img) for img in path]
path = glob.glob("t_kagit/*.JPG") #or jpg


cv_image = [read_img(test, img) for img in path]
path = glob.glob("t_kagit/*.jpeg") #or jpg


cv_image = [read_img(test, img) for img in path]
path = glob.glob("t_metal/*.JPG") #or jpg

cv_image = [read_img(test, img) for img in path]
path = glob.glob("t_metal/*.jpeg") #or jpg


cv_image = [read_img(test, img) for img in path]
path = glob.glob("t_plastik/*.JPG") #or jpg


cv_image = [read_img(test, img) for img in path]
path = glob.glob("t_plastik/*.jpeg") #or jpg


cv_image = [read_img(test, img) for img in path]
y=np.array(test)
#--------------------
#
#--------------------processing train test data
x = (x / 255) - 0.5
#x = np.expand_dims(x, axis=3)

y = (y / 255) - 0.5
#y = np.expand_dims(y, axis=3)
#---------------------
#
#----------------------train test label son hali
import pandas as pd
x_train=x
y_train=y
x_test=pd.read_excel('label.xlsx')
x_test=np.array(x_test)
x_test=x_test.reshape(1630,)
y_test=pd.read_excel('label_2.xlsx')
y_test=np.array(y_test)
y_test=y_test.reshape(321,)

#--------------------
#
#-----------------------CNN
from keras.utils import to_categorical
#Evrişimli Sinir Ağı Mimarisini Oluşturma
model = Sequential()

#1. evrişim katmanı
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2. Evrişim katmanı
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#3. Evrişim katmanı
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

# Tam bağlantı katmanı
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)


model.fit(x_train,
          to_categorical(x_test),
          nb_epoch = 1,
          validation_data=(y_train, to_categorical(y_test)))

#------------------------------------------------
#
#score = model.evaluate(x_test,y_test,verbose=0)
#print('Test Score = ',score[0])
#print('Test Accuracy =', score[1])
#
from keras.models import load_model
#------------------Model save
model.save('my_model.h5') 
#-----------------------------
#
#-----------------Model load
model = load_model('my_model.h5')
#-----------------------------
#
#------------------------------TEST IMAGE 
image = cv2.imread('b_c_1.JPG') 
test_den_img=cv2.resize(image,(64,64))
test_den_img = np.expand_dims(test_den_img, axis = 0)
test_den_img = (test_den_img / 255) - 0.5
custom = model.predict(test_den_img)
test_den_result=np.argmax(custom)
#-------------------------------------
#
#
##--------------------------------TEST WEBCAM
#camera = cv2.VideoCapture(0)# Video çekmeye başla
#return_value,image = camera.read()# İlk fotğrafı al
#cv2.imwrite('test.JPG',image)#Kaydet
#camera.release()# ?
#cv2.destroyAllWindows()# Tüm ekranları kapat
#test_den_img=cv2.resize(image,(64,64))
#test_den_img = np.expand_dims(test_den_img, axis = 0)
#test_den_img = (test_den_img / 255) - 0.5
#custom = model.predict(test_den_img)
#
#test_den_result=np.argmax(custom)
#
#if test_den_result == 0:
#    print("Atık =  CAM")
#elif test_den_result == 1:
#    print("Atık =  KAĞIT")
#elif test_den_result == 2:
#    print("Atık =  METAL")
#elif test_den_result == 3:
#    print("Atık =  PLASTİK")
#    
#
##0 CAM 1 KAGİT 2 METAL 3 PLASTİK
##------------------------------------
