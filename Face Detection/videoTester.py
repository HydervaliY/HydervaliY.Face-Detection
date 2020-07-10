import os
import cv2
import numpy as np
import faceRecognition as fr
import random

#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name={0:"Hyder RA1711003010174",1:"Hari RA1711003010272",2:"Iyer RA1711003010094",3:"Kevin RA1711003010095",
4:"Dilip RA1711003010274",5:"Chandu RA1711003010374"}
name1={0:"Innocent",1:"Studios",2:"Talented",3:"Celeb",4:"Actor",5:"machaa"}

cap=cv2.VideoCapture(0)
c=0
while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in faces_detected:
      l=[(0,255,100),(255,0,100),(0,100,255)]
      cv2.rectangle(test_img,(x,y),(x+w+2,y+h),random.choice(l),thickness=2)

    resized_img = cv2.resize(test_img, (800, 700))
    cv2.imshow('face detection Tutorial ',resized_img)
    cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        predicted_name1=name1[label]
        if confidence < 55:#If confidence less than  then don't priqnt predicted face text on screen
           c=c+1
           fr.put_text(test_img,predicted_name,x,y)
           fr.put_text(test_img,predicted_name1+"  "+str(c),x,y-24)
           if c>2:
               fr.put_text(test_img,"Duplicate",x,y-50)
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition',resized_img)
    if cv2.waitKey(10) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows

