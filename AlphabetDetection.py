# IMPORT MODULES
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

# LOAD THE DATA
x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J',"K",'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)
# SPLIT THE DATA

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 2500,train_size = 7500,random_state = 26)

# SCALE THE FEATURES

x_train_scaled = x_train/255
x_test_scaled = x_test/255

# CREATE THE CLASSIFIER

cls = LogisticRegression(solver = 'saga',multi_class='multinomial').fit(x_train_scaled,y_train)

# CALCULATE THE ACCURACY 

y_pred = cls.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# STARTING THE CAMERA

cap = cv2.VideoCapture(0)

while (True):

    # CAPTURE FRAME BY FRAME 
    ret,frame = cap.read()

    # OUR OPERATIONS ON THE FRAME COMES HERE
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # DRAWING A BOX IN THE CENTER OF THE VIDEO

    height,width = gray.shape
    upper_left = (int(width/2-56),int(height/2-56))
    bottom_right = (int(width/2+56),int(height/2+56))

    cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)

    # TO ONLY CONSIDER THE AREA INSIDE THE BOX FOR DETECTING THE DIGIT
    # ROI = region of interest
    roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

    # CONVERTING CV2 IMAGE TO PIL FORMAT
    im_pil = Image.fromarray(roi)

    # CONVERT PIXEL IMAGE TO GRAY SCALE 
    image_bw = im_pil.convert('L')

    # resize the image
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)

    # INVERT THE IMAGE
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20

    # CONVERTING TO SCALER QUANTITY
    min_pixel = np.percentile(image_bw_resized_inverted,pixel_filter)

    # USING CLIP TO LIMIT THE COLOR VALUES BETWEEN 0 TO 255
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel,0,255)
    max_pixel = np.max(image_bw_resized_inverted)

    # CONVERTING THE IMAGE IN AN ARRAY
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

    # CREATING A TEST SAMPLE AND MAKING THE PREDICTION
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_prediction = cls.predict(test_sample)
    print("PREDICTED CLASS IS : ",test_prediction)

    # DISPLAY THE RESULTING FRAME AND OFF THE CAMERA
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()






