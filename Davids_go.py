# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:29:06 2020

@author: David Marples
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def DShowImage(img):
    plt.figure()
    plt.imshow(img,cmap=plt.cm.binary)
    plt.show()
 
def ToMono(img,threshold):
    with np.nditer(img,op_flags=['readwrite']) as it:
        for x in it:
            if x>threshold:
                x[...]=1.0
            else:
                x[...]=0    

def FindFeatures(img):
    #Feature characteristics:
    #Total labelling, within the whole 28x28 grid(1)
    #Width of character (xmax-xmin: 1 number)
    #Height of character (ymax-ymin: 1 number)
    #labelling in top, middle, and bottom thirds (3)
    #labelling in left, centre, and right thirds (3)
     #two loops, with top, bottom,width, and whether closed at top and bottom (10)
    Featuredat=np.zeros(19)
    rows=np.zeros(28)
    columns=np.zeros(28)
    lines=np.zeros((28,5))
    minx=30
    maxx=0
    miny=30
    maxy=0
    #Ignore sections of the image with nothing in
    for x in range(28):
        rows[x]=img[x].sum()
        columns[x]=img[:,x].sum()
        for y in range(28):
            if img[y,x]==1:
                if y<miny:
                    miny=y
                if y>maxy:
                    maxy=y
            if img[x,y]==1:
                if y<minx:
                    minx=y
                if y>maxx:
                    maxx=y
    Featuredat[0]=rows.sum()
    Featuredat[1]=maxx-minx+1
    Featuredat[2]=maxy-miny+1
    #Now we calculate the left,middle, right weights, then top/middle/bottom
    xo3=(maxy-miny+1)/3
    for x in range(miny,maxy+1):
        if x<=miny+xo3:
            Featuredat[3]+=rows[x]
        elif x<maxy-xo3:
            Featuredat[4]+=rows[x]
        else:
            Featuredat[5]+=rows[x]
    
    xo3=(maxx-minx+1)/3  
    for x in range(minx,maxx+1):
        if x<=minx+xo3:
            Featuredat[6]+=columns[x]
        elif x<maxx-xo3:
            Featuredat[7]+=columns[x]
        else:
            Featuredat[8]+=columns[x]
    
    #Now work out the "loops" and their characteristics
    nloops=0
    maxgwidth=0
    inloop=False
    tclosed=0
    bclosed=0
    ltop=0
    lbottom=0
    lend=0
    rend=0
    for y in range(miny,maxy+1):
        lstart=-1
        rstart=-1
        gwidth=0
        for x in range(minx,maxx+1):
            if img[y,x]==1:
                if lstart==-1:
                    lstart=x
                    lend=x
                else:
                    if gwidth<1:
                        lend+=1
                        gwidth=0
                    else:
                        if rstart==-1:
                            rstart=x
                            rend=x
                        else: 
                            rend+=1
            else:
                #value is 0: we are in a gap
                if lstart>0 and rstart==-1:
                    gwidth+=1
                    
        #OK, finished gathering data from the line: let's save and process it
        #Need to save this data so we can refer back to it from subsequent lines
        #Could probably hold just one line if we moved it to the end of the processing...
        lines[y,0]=lstart
        lines[y,1]=lend
        lines[y,2]=gwidth
        lines[y,3]=rstart
        lines[y,4]=rend
                        
        if rstart>0:
            #There's a gap between (at least) two lines, so we must be in a loop
            if inloop:
                if gwidth>maxgwidth:
                    maxgwidth=gwidth
            else:
                inloop=True
                nloops+=1
                if img[y-1,int((rstart+lend)/2)]==1:
                    tclosed=1
                ltop=y
        else:
            #rstart=0: have we just finished processing a loop?
            if inloop:
                inloop=False
                if img[y,int((lines[y-1,3]+lines[y-1,1])/2)]==1:
                    bclosed=1
                lbottom=y
                if lbottom-ltop<3:
                    #loop is no more than 2 deep, and is probably an artifact
                    nloops-=1
                    tclosed=0
                    bclosed=0
                    gwidth=0
                    maxgwidth=0
                else:
                    #we'll count this loop, if it's one of the first 2
                    #print("Loop detected!",nloops)
                    if nloops<3:
                        Featuredat[4+5*nloops]=ltop-miny
                        Featuredat[5+5*nloops]=lbottom-miny
                        Featuredat[6+5*nloops]=maxgwidth
                        Featuredat[7+5*nloops]=tclosed
                        Featuredat[8+5*nloops]=bclosed
    return Featuredat                 
    
    
def PreProcess(numimages):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    dtrain=np.zeros((numimages,19))
    dlabels=np.zeros(numimages)

    for tin in range(0,numimages):
        ti=train_images[tin].copy()
        ToMono(ti,0.1)
        fd=FindFeatures(ti)
        dtrain[tin]=fd
        dlabels[tin]=train_labels[tin]
        np.save("DImageData.npy",dtrain)
        np.save("DLabelData.npy",dlabels)





class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']

#train_images = train_images / 255.0

#numimages=50000
#PreProcess(numimages)

dtrain=np.load("DImageData50k.npy")
dlabels=np.load("DLabelData50k.npy")



model = keras.Sequential([
    keras.layers.Input(19),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(dtrain, dlabels,batch_size=100,epochs=20)

