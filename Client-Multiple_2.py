import socket
from threading import Thread
import sys 

import cv2
import numpy as np
from shapely import geometry
from shapely.geometry import box
from shapely.geometry import Point
import time

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# import geopandas as gpd
from shapely.geometry import Point
import pygame
pygame.mixer.init()
# my_sound = pygame.mixer.Sound('kick.wav')
# my_sound.play()
# my_sound.set_volume(0.5)

import numpy as np
def distance2D(a,b):
    dist = np.linalg.norm(np.array(a)*1000-np.array(b)*1000)
    return dist

# distance2D(Lwrist,lastLWrist)

def avg(lst):
    return sum(lst) / len(lst)

import pygame
pygame.mixer.init()
my_sound = pygame.mixer.Sound('kick.wav')
# my_sound.play()
my_sound.set_volume(0.5)

import threading
from playsound import playsound



def kick(vol=1):
    my_sound = pygame.mixer.Sound('kick.wav')
    my_sound.set_volume(vol)
    my_sound.play()
    
    
def snare(vol=1):   
    my_sound = pygame.mixer.Sound('snare.wav')
    my_sound.set_volume(vol)
    my_sound.play()
    
def crash(vol=1): 
    my_sound = pygame.mixer.Sound('crash.wav')
    my_sound.set_volume(vol)
    my_sound.play()
    
def hihat(vol=1):   
    my_sound = pygame.mixer.Sound('hihat.wav')
    my_sound.set_volume(vol)
    my_sound.play()


    
instrumentR1=instrumentL1=0
instrumentR2=instrumentL2=0

lastLWrist=[0,0]
lastRWrist=[0,0]

lWSpeed=[0]*5
rWSpeed=[0]*5

pp=jj=255
dis=30

adjx1=adjy1=adjx2=adjy2=0


setup=False

# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
start_point1 = (0, 0)
end_point1 = (100, 100)
color1 = (255, 0, 0)

start_point2 = (640-100, 0)
end_point2= (640, 100)
color2 = (255, 0, 0)

thickness = 2
fingDistance=[0]*10
setupbox1 = box(0,0,100,100)
setupbox2 = box(640-100,0,640,100)
font = cv2.FONT_HERSHEY_SIMPLEX



ClientSocket = socket.socket()
host = 'localhost'
port = 1233

print('Waiting for connection')
try:
    ClientSocket.connect((host, port))
except socket.error as e:
    print(str(e))

# Response = ClientSocket.recv(1024)
# while True:
#     Input = input('Say Something: ')
#     ClientSocket.send(str.encode(Input))
#     Response = ClientSocket.recv(1024)
#     print(Response.decode('utf-8'))

# ClientSocket.close()


def recv():
    global name
    while True:
        data = s.recv(1024).decode()
        if not data: sys.exit(0)
        data= data[data.find(',')+1:]
        print (data)
        if(data=="kick"):
            kick()
        if(data =="snare"):
            snare()
        if(data =="crash"):
            crash()
        if(data=="hihat"):
            hihat()
        
         
# name = input("Enter your name: ")
name= "Sutirtha"
s = socket.socket()
s.connect((host,port))
Thread(target=recv).start()
# while 1:
    # message = input("Message: ")
    # s.send("{}: {}".format(name, message).encode('utf-8'))
    # s.send("{}".format(message).encode('utf-8'))
red_img  = np.full((480, 640, 3), (0,0,255), np.uint8)
blue_img  = np.full((480, 640, 3), (255,0,0), np.uint8)

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5) as holistic:
    while True:
        ret, background = cap.read()
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        background = cv2.flip(background,1)
        results = holistic.process(background)
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        background = cv2.rectangle(background, start_point1, end_point1, color1, thickness)
        background = cv2.rectangle(background, start_point2, end_point2, color2, thickness)
        yscale,xscale= background.shape[:-1]
        img1 = cv2.imread('p1.png')
        resized1 = cv2.resize(img1, (dis,dis), interpolation = cv2.INTER_AREA)
        img2 = cv2.imread('p2.png')
        resized2 = cv2.resize(img2, (dis,dis), interpolation = cv2.INTER_AREA)



        try:
            if results.pose_landmarks:
                ########## Draw face landmarks
#                 mp_drawing.draw_landmarks(background, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                ##########Right hand
                # mp_drawing.draw_landmarks(background, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                ########## Left Hand
                # mp_drawing.draw_landmarks(background, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                ########## Pose Detections
                # mp_drawing.draw_landmarks(background, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
                nose =  [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y]
                
                # play with foot
                Lwrist= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].y]
                Rwrist= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].y]
                
                # with arms
                Rwrist= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y]
                Lwrist= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].y]
                
                RindexFingure= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x*xscale, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y*yscale]
                LindexFingure= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark. LEFT_INDEX].x*xscale, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].y*yscale]
                
                # with hips
                Ref1=[results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x*xscale,  results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y*yscale]
                Ref2=[results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x*xscale, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y*yscale]

                # for legs
                # Ref1= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].x*xscale,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].y*yscale]
                # Ref2= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].x*xscale,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].y*yscale]
                


                LeftEar = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x* xscale,   results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y*yscale]
                RightEar= [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x* xscale,  results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y*yscale]

                x1=Ref1[0]+adjx1
                y1=Ref1[1]-dis+adjy1
                x2=Ref1[0]+dis+adjx1
                y2=Ref1[1]+adjy1
                box1 = box(x1,y1,x2,y2)
                background[int(y1):int(y2),int(x1):int(x2)] = resized1
                
                x3=Ref2[0]-dis+adjx2
                y3=Ref2[1]-dis+adjy2
                x4=Ref2[0]+adjx2
                y4=Ref2[1]+adjy2
                box2 = box(x3,y3,x4,y4)
                background[int(y3):int(y4),int(x3):int(x4)] = resized2
                
                if(setupbox1.contains(Point(RindexFingure))):
                    color1 = (255, 255, 0)
                    adjx1=int(LindexFingure[0]-Ref1[0])
                    adjy1=int(LindexFingure[1]-Ref1[1])
                elif(setupbox2.contains(Point(LindexFingure))):
                    color2 = (255, 255, 0)
                    adjx2=int(RindexFingure[0]-Ref2[0])
                    adjy2=int(RindexFingure[1]-Ref2[1])
                else:
                    color1 = (255, 0, 0)
                    color2 = (255, 0, 0)
                    lWSpeed.append(round(distance2D(Lwrist,lastLWrist)))
                    rWSpeed.append(round(distance2D(Rwrist,lastRWrist)))

                    lWSpeed=lWSpeed[-5:]
                    rWSpeed=rWSpeed[-5:]
                    
                    # instrument 1
                    speed=15

                    if box1.contains(Point(Lwrist[0]*xscale,Lwrist[1]*yscale)):
                        jj=100
                        if(sum(lWSpeed)/5>speed and instrumentL1==0):
                            hihat()
                            s.send("{},{}".format(name,"hihat").encode('utf-8'))
                            instrumentL1=1
                            # resized1  = cv2.addWeighted(resized1, 0.8, red_img, 0.2, 0)
                            background  = cv2.addWeighted(background, 0.8, red_img, 0.2, 0)
                    else:
                        instrumentL1=0
                        jj=255

                    if box1.contains(Point(Rwrist[0]*xscale,Rwrist[1]*yscale)):
                        pp=100
                        if(sum(rWSpeed)/5>speed and instrumentR1==0):
                            hihat()
                            s.send("{},{}".format(name,"hihat").encode('utf-8'))
                            instrumentR1=1
                            # resized1  = cv2.addWeighted(resized1, 0.8, red_img, 0.2, 0)
                            background  = cv2.addWeighted(background, 0.8, red_img, 0.2, 0)
                    else:
                        instrumentR1=0 
                        pp=255


                    speed=15
                    # instrument 2
                    if box2.contains(Point(Lwrist[0]*xscale,Lwrist[1]*yscale)):
                        jj=88
                        if(sum(lWSpeed)/5>speed and instrumentL2==0):
                            kick()
                            s.send("{},{}".format(name,"kick").encode('utf-8'))
                            instrumentL2=1
                            # resized2  = cv2.addWeighted(resized2, 0.8, blue_img, 0.2, 0)
                            background  = cv2.addWeighted(background, 0.8, blue_img, 0.2, 0)
                    else:
                        instrumentL2=0
                        jj=255

                    if box2.contains(Point(Rwrist[0]*xscale,Rwrist[1]*yscale)):
                        pp=200
                        if(sum(rWSpeed)/5>speed and instrumentR2==0):
                            kick()
                            s.send("{},{}".format(name,"kick").encode('utf-8'))
                            instrumentR2=1
                            # resized2  = cv2.addWeighted(resized2, 0.8, blue_img, 0.2, 0)
                            background  = cv2.addWeighted(background, 0.8, blue_img, 0.2, 0)
                    else:
                        instrumentR2=0 
                        pp=255

                    dis=round(distance2D(LeftEar,RightEar)/1000)+20 +20  # Distance Factor

                    # cv2.putText(background, str(sum(lWSpeed)/5) , 
                    #                    tuple(np.multiply(Rwrist, [xscale, yscale]).astype(int)), 
                    #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, jj, 255), 2, cv2.LINE_AA)


                    yy= sum(lWSpeed)/5
                    yy= sum(rWSpeed)/5


                    lastLWrist=Lwrist
                    lastRWrist=Rwrist

 

        except Exception as e: 
            pass
            # print(e)


        cv2.imshow('DrumClient',background)
        k = cv2.waitKey(10)
        # Press q to break
        if k == ord('q'):
            break

#  Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
