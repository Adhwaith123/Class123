import csv
import pandas as pd 
import random
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import Pil.ImageOps
import os,time
import cv2
from sklearn.metrics import accuracy_score
x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=["0","1","2","3","4","5","6","7","8","9"]
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=1,train_size=7500,test_size=2500)
xtrainScaled=xtrain/255.0
xtestScaled=xtest/255.0
lr=LogisticRegression(solver="saga",muti_class="multinomial").fit(xtrainScaled,ytrain)
yPredict=lr.predict(xtestScaled)
acurracy=accuracy_score(ytest,yPredict)
print(accuracy)
capt=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=capt.read()
        height,width=gray.shape
        upperLeft=(int(width/2-56),int(height/2-56))
        bottomRight=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi=gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]
        im_pil=Image.fromarray(roi)
        imagebw=im_pil.convert("L")
        imagebwResize=imagebw.resize(("28,28"),Image.ANTIALIAS)
        imageInverted=PIL.ImageOps(imagebwResize)
        pixelfiltered=20
        minpixel=np.percentile(imageInverted,pixelfiltered)
        imageInvertedScale=np.clip(imageInverted-minpixel,0,255)
        maxpixel=np.max(imageInverted)
        imageInvertedScale=np.asarray(imageInvertedScale1)/maxpixel
        testSample=np.array(imageInvertedScale).reshape(1,784)
        testPredict=lr.predict(testSample)
        print("predictedClasses:",testPredict)
        cv2.imshow("frame",gray)
        if (cv2.waitKey(1)):
            break
    except Exception as e:
        pass     

    
    capt.release()
    cv2.destroyAllWindows