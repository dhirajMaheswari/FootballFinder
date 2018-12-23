'''
Created on Aug 31, 2018

@author: Dhiraj

this code uses the custom build haar-like cascade file to detect balls in images
'''

import cv2
import numpy as np
import os, time

#imgSrc = "pictures/balls/1.bmp"

# img = cv2.imread(imgSrc) #use cv2.IMREAD_GRAYSCALE as second argument to load in grayscale
# # img is a numpy array, with dimensions (h,w,colorchannels)
# print(img.shape, img.shape[0], img.shape[1])


impath = os.path.curdir + r'/pictures/balls/' # path to image files

#ball_cascade = cv2.CascadeClassifier(
#        'haar_cascade_files/myfootballdetector.xml') # load the football cascade file
 
ball_cascade = cv2.CascadeClassifier(
        'myfootballdetector.xml') # load the football cascade file
 
if ball_cascade.empty():
    raise IOError('Unable to load the football cascade classifier xml file')
 
''' iterate over all the files in the specified path, assuming they are all images'''

for f in os.listdir(impath):
    imgSrc = impath + f # image to check for football
    #print(imgSrc)

    img = cv2.imread(imgSrc) # read the image
    
    cv2.imshow('looking for footballs', img)

#     cv2.imshow('balls',img)
#     cv2.waitKey(1000)
 
    scaling_factor =1.0
    img = cv2.resize(img, None, 
          fx=scaling_factor, fy=scaling_factor, 
          interpolation=cv2.INTER_AREA)
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    balls = ball_cascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5
        ) # look for the balls in the image
    print ("In %s, Found %d balls." %(imgSrc, len(balls)))
    
#   
    if len(balls) == 0:
        cv2.imshow('cannot find ball in this', img)

    for (x,y,w,h) in balls:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] # these are not relevant in this example here.
        roi_color = img[y:y+h, x:x+w] # these can be used to create more images
      
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
      
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (x,y)
        fontScale              = 1
        fontColor              = (0,0,255)
        lineType               = 2
  
        cv2.putText(img,'ball', 
                    topLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
      
        cv2.imshow('With football detected', img)
 
 
        cv2.waitKey(1000) # wait for 1000 milliseconds, 
        
#     if cv2.waitKey(32) & 0xFF == 32:
#         break




 
#  cv2.destroyAllWindows()