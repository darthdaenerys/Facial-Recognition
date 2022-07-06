import cv2
import numpy as np
import face_recognition as FR
import time
import tkinter as tk 
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
print('Input your image file')
file_path = filedialog.askopenfilename()
print('Enter name')
name=input()

width=1280
height=720

camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,width)
camera.set(cv2.CAP_PROP_FPS,30)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

Face=FR.load_image_file(file_path)
FaceEncode=FR.face_encodings(Face)[0]

knownNames=[name]
knownFaceEncodes=[FaceEncode]

flag=0
fps=2
dt=2
t1=time.time()
while True:
    _,frame=camera.read()
    unknownFaceLoc=FR.face_locations(frame)
    unknownFaceRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    unknownFacesEncodes=FR.face_encodings(unknownFaceRGB,unknownFaceLoc)
    for unknownFace,unknownFaceEncode in zip(unknownFaceLoc,unknownFacesEncodes):
        top,right,bottom,left=unknownFace
        match=FR.compare_faces(knownFaceEncodes,unknownFaceEncode)
        name='Unknown'
        color=(0,0,255)
        for i,stat in enumerate(match):
            if stat:
                name=knownNames[i]
                color=(0,255,0)
                break
        cv2.rectangle(frame,(left,top),(right,bottom),color,2)
        cv2.rectangle(frame,(left,bottom),(right,bottom+40),color,-1)
        cv2.putText(frame,name,(left+10,bottom+30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
    dt=time.time()-t1
    t1=time.time()
    fps=int(1//dt)
    cv2.putText(frame,str(fps)+' FPS',(50,80),cv2.FONT_HERSHEY_DUPLEX,1,(145,42,98),2)
    cv2.imshow('Face Recognition',frame)
    if cv2.waitKey(5) & 0xff==ord('q'):
        break
camera.release()