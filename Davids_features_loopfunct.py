# -*- coding: utf-8 -*-
""" Feature-based Digit Recognition

    This is an attempt at an "intelligent design" routine for hadnwritten
    digit recogntion. The goal is to extract specific features, such as loops,
    the number of horizontal and vertical lines at particular points, etc,
    and use them to identify the handwritten digits of the MNIST dataset.
    
    One branch of the code will feed the features vector to a neural net:
    the hope is that effectively the feature extraction forms the "hidden layers",
    and such a network can be just 2 layers thick.
    
    The other approach is to develop a set of weightings manually, as an array
    that can be simply multiplied with the vector to give likelihoods for the 
    different digits. This will be partly informed guesswork, and partly 
    guided training.

    @author: David Marples
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def DShowImage(img):
    """Uses pyplot to display an MNIST style image"""    
    plt.figure()
    plt.imshow(img,cmap=plt.cm.binary)
    plt.show()

def FindLoop(img,lines,scanstart,scanend,pixstart,pixend,vertical):
    """Looks for a 'loop' in either vertical or horizontal plane.
    
       Basically scans through the (specified part of) the image, and where it 
       finds two regions of 'line', it considers the space between to be part
       of a loop. When it enters a loop, it decides whether it was closed before
       it, and when it leaves it decides whether it is close after it.
       
       As parameters it takes 
           1) an image of unspecified size, which is assumed to be B&W, with 
              'set' pixels having the value 1.
           2) an array for line data (essentially a way of returning the data)
              I know this could / should be improved...
           3) Start and end values for the scan,  
           4) start and end values for the part of the image to scan across
           5) 'vertical' is a boolean flag determining the direction of scan
       
       It returns the start and end positions within the scan range, the maximum
       width of the loop, and two flags (1 for closed, at top and bottom)."""
    outcome = np.zeros(5)
    maxgwidth = 0
    inloop = False
    tclosed = 0
    bclosed = 0
    ltop = 0
    lbottom = 0
    lend = 0
    rend = 0
    for x in range(int(scanstart),int(scanend+1)):
        lstart = -1
        rstart = -1
        gwidth = 0
        for y in range(int(pixstart),int(pixend+1)):
            if (img[x,y] == 1 and not vertical) or (img[y,x] == 1 and vertical):
                if lstart == -1:
                    lstart = y
                    lend = y
                else:
                    if gwidth < 1:
                        lend += 1
                        gwidth = 0
                    else:
                        if rstart == -1:
                            rstart = y
                            rend = y
                        else: 
                            rend += 1
            else:
                #value is 0: we are in a gap
                if lstart > 0 and rstart == -1:
                    gwidth += 1
                    
        """OK, finished gathering data from the line: let's save and process it
           Need to save this data so we can refer back to it from subsequent lines
           Data passed back to feature finder to use for maxlinelength"""
        
        maxlinelength = lend - lstart + 1
        if rstart > lend and (rend - rstart + 1) > maxlinelength:
            maxlinelength = (rend - rstart + 1)
            
        lines[x,0] = lstart
        lines[x,1] = lend
        lines[x,2] = gwidth
        lines[x,3] = rstart
        lines[x,4] = rend
        lines[x,5] = maxlinelength
                        
        if rstart > 0:
            #There's a gap between (at least) two lines, so we must be in a loop
            if inloop:
                if lstart < lines[x - 1,1] + 2 and lend > lines[x - 1,3] - 2:
                    inloop = False
                    bclosed = 1
                    lbottom = x
                    if lbottom-ltop<2:
                        #loop is no more than 1 deep, and is probably an artifact
                        tclosed = 0
                        bclosed = 0
                        gwidth = 0
                        maxgwidth = 0
                
                elif gwidth > maxgwidth:
                    maxgwidth = gwidth
            else:
                inloop = True
                if lend > lines[x-1,0] - 2 and rstart<lines[x-1,1] + 2:
                    tclosed = 1
                ltop = x
        else:
            #rstart=0: have we just finished processing a loop?
            if inloop:
                inloop = False
                if lstart < lines[x-1,1] + 2 and lend > lines[x-1,3] - 2:
                    bclosed = 1
                lbottom = x
                if lbottom - ltop < 3:
                    #loop is no more than 2 deep, and is probably an artifact
                    tclosed = 0
                    bclosed = 0
                    gwidth = 0
                    maxgwidth = 0
                else:
                    outcome[0] = ltop - scanstart
                    outcome[1] = lbottom - scanstart
                    outcome[2] = maxgwidth
                    outcome[3] = tclosed
                    outcome[4] = bclosed
                    return outcome
    #Will get here (and return 0's) if there is never a loop
    return outcome

 
def FindFeatures(img):
    """ Find relevant features of the image, and store in a vector.
    
    Assumes a 28 x 28 image, with pixels either 0 or 1.
    
    Feature characteristics:
    Total labelling, within the whole 28x28 grid(1 (0))
    Width of character (xmax-xmin: 1 number (1))
    Height of character (ymax-ymin: 1 number (2))
    labelling in top, middle, and bottom thirds (3 (3 - 5))
    labelling in left, centre, and right thirds (3 (6 - 8))
    two loops, with top, bottom,width, and whether closed at top and bottom (10 (9 - 13 and 14 - 18))
    Then longest lines in top, middle, and bottom thirds (3 (19-21))
    And longest lines in left, middle and right thirds (3 (22-24))
    And number of "lines" in left middle and right thirds (3 (25-27)
    And loop data for top half (5 (28 - 32))
    And loop data for bottom half (5 (33 - 37))
    """
    Featuredat = np.zeros(38)
    rows = np.zeros(28)
    columns = np.zeros(28)
    lines = np.zeros((28,6))
    """Find the bounding box of the actual number. """
    itemindex = np.array(np.where(img > 0.1))
    (minx, miny) = int(itemindex[1].min()), int(itemindex[0].min())
    (maxx, maxy) = int(itemindex[1].max()), int(itemindex[0].max())
    """Sum each row and column """
    for x in range(28):
        rows[x] = img[x,:].sum()
        columns[x] = img[:,x].sum()
    
    Featuredat[0] = img.sum()
    Featuredat[1] = maxx - minx + 1
    Featuredat[2] = maxy - miny + 1
    
    #Now work out the "loops" and their characteristics
    """"Look for curves facing left or right, in the upper and lower 
        parts of the digit - particularly for 2, 3, 5"""
    Featuredat[28:33] = FindLoop(img,lines,minx,maxx,miny,int((miny+2*maxy)/3),True)
    Featuredat[33:38] = FindLoop(img,lines,minx,maxx,int((2*miny+maxy)/3),maxy,True)
    """Look for curves from top or bottom (eg top of 2, or 9 or 6).
       These ones need to come second, since the data in lines is needed later"""
    Featuredat[9:14] = FindLoop(img,lines,miny,maxy,minx,maxx,False)
    if Featuredat[10] > 0:
        Featuredat[14:19] = FindLoop(img,lines,Featuredat[10]+miny,maxy,minx,maxx,False)
      
    
    #Now we calculate the left,middle, right weights, then top/middle/bottom
    xo3 = (maxy - miny + 1) / 3
    for x in range(miny,maxy+1):
        
            
        if x <= miny + xo3:
            Featuredat[3] += rows[x]
            if lines[x,5] > Featuredat[19]:
                Featuredat[19] = lines[x,5]
        elif x < maxy - xo3:
            Featuredat[4] += rows[x]
            if lines[x,5] > Featuredat[20]:
                Featuredat[20] = lines[x,5]
        
        else:
            Featuredat[5] += rows[x]
            if lines[x,5] > Featuredat[21]:
                Featuredat[21] = lines[x,5]
        
    
    xo3 = (maxx - minx + 1) / 3  
    for x in range(minx,maxx+1):
        vstart = 0
        maxvline = 0
        nlines = 0
        
        for y in range(miny,maxy+1):
            if img[y,x] == 0 and vstart > 0:
                #ended a line that started at vstart
                if maxvline < y-vstart:
                    maxvline = y-vstart
                vstart=0
            if img[y,x] == 1:
                if vstart == 0:
                    vstart = y
                    nlines += 1
            
            
        if x <= minx + xo3:
            Featuredat[6] += columns[x]
            Featuredat[25] += nlines / xo3
            if maxvline > Featuredat[22]:
                Featuredat[22] = maxvline
                
        elif x < maxx-xo3:
            Featuredat[7] += columns[x]
            Featuredat[26] += nlines / xo3
            if maxvline > Featuredat[23]:
                Featuredat[23] = maxvline
        else:
            Featuredat[8] += columns[x]
            Featuredat[27] += nlines / xo3
            if maxvline > Featuredat[24]:
                Featuredat[24] = maxvline
     

           
    return Featuredat

def Interpret(fd):
    """ Now we use the features etc to work out which letter it is likely to be.
    
    This version works by looking for evidence for each digit.
    
    Feature characteristics:
    Total labelling, within the whole 28x28 grid(1 (0))
    Width of character (xmax-xmin: 1 number (1))
    Height of character (ymax-ymin: 1 number (2))
    labelling in top, middle, and bottom thirds (3 (3 - 5))
    labelling in left, centre, and right thirds (3 (6 - 8))
    two loops, with top, bottom,width, and whether closed at top and bottom (10 (9 - 13 and 14 - 18))
    Then longest lines in top, middle, and bottom thirds (3 (19-21))
    And longest lines in left, middle and right thirds (3 (22-24))
    And number of "lines" in left middle and right thirds (3 (25-27)
    And loop data for top half (5 (28 - 32))
    And loop data for bottom half (5 (33 - 37))"""
    
    predictions=np.zeros(11)
    predicted=10
    
    midpt = fd[2]//2
    topthird = fd[2] // 3
    botthird = fd[2]*2//3
    
    #Let's look for evidence of a zero
    if fd[9] < midpt and fd[10] > botthird:
        predictions[0] = 0.9
    if fd[22]<7:
        predictions[0] -= 0.1
    
    #Let's look for evidence of a one
    if fd[19] < 7 and fd[20] < 7 and fd[21] < 7:
        predictions[1] = 0.9
        
    #Let's look for evidence of a two
    if fd[31] == 0 and fd[32] == 1:
        predictions[2] = 0.3
    if fd[12] == 1 and fd[13] == 0:
        predictions[2] += 0.2
    if  fd[36] == 1 and fd[37] == 0 and fd[33] < midpt:
        predictions[2] += 0.2
    if fd[21] > 10:
        predictions[2] += 0.2
   
    #Let's look for evidence of a three
    if fd[31] == 0 and fd[32] == 1:
        predictions[3] = 0.4
    if  fd[36] == 1 and fd[37] == 1:
        predictions[3] += 0.4
    if  fd[36] == 1 and fd[37] == 1:
        predictions[3] += 0.4
    if fd[19] > 8 and fd[20] > 8 and fd[21] > 8:
        predictions[3] += 0.2
    if fd[8] - fd[6] > 5:
        predictions[3] += 0.2
    
    #Look for evidence of a four:
    if fd[12] == 0 and fd[13] == 1:
        predictions[4] = 0.5
    if fd[12] == 1 and fd[13] == 1 and fd[20] > 12:
        predictions[4] = 0.3
    if fd[19] < 8:
        predictions[4] += 0.2
    if fd[20] > 10:
        predictions[4] += 0.2
    if fd[21] < 6:
        predictions[4] += 0.4
        
    #Look for evidence of a 5
    if fd[31] == 1 and fd[32] == 0:
        predictions[5] = 0.3
    if  fd[36] == 0 and fd[37] == 1:
        predictions[5] += 0.3
    if fd[6] - fd[8] > 5:
        predictions[5] += 0.2
    if fd[19] > 8:
        predictions[5] += 0.2
        
    #look for evidence of a 6
    if fd[13] == 1 and fd[9] > topthird:
        predictions[6] = 0.5
    if fd[19] < 8:
        predictions[6] += 0.5
    if fd[31] == 1 and fd[32] == 0:
        predictions[6] += 0.2
    if fd[36] == 1 and fd[37] == 1 and fd[19] < 8:
        predictions[6] += 0.5
    
    #Look for evidence of a seven
    if fd[19] > 8 and fd[21] < 7:
        predictions[7] = 0.8
    if fd[12] == 1 and fd[13] == 1:
        predictions[7] -= 0.3
        
    #Look for evidence of an 8
    if fd[17] == 1 and fd[18] == 1:
        predictions[8] = 1.8
    if fd[9] > topthird and fd[12] == 1 and fd[13] == 1 and (fd[31] == 1 or fd [32] == 1) and fd[19] > 7:
        predictions[8] = 1.8
    if fd[9] < midpt and fd[12] == 1 and fd[13] == 1 and fd[5] / fd[3] > 1:
        predictions[8] = 0.8
    
    
    #Look for evidence of a nine
    if fd[10] < botthird and fd[12] == 1 and fd[13] == 1 and fd[21] < 10:
        predictions[9] = 1.0
    if fd[31] == 1 and fd[32] == 1 and fd[21] < 7:
        predictions[9] = 0.9
    if fd[19] - fd[20] > -3:
        predictions[9] += 0.2
    if fd[17] == 1 and fd[18] == 1:
        predictions[9] -= 0.8
        
    """Total labelling, within the whole 28x28 grid(1 (0))
    Width of character (xmax-xmin: 1 number (1))
    Height of character (ymax-ymin: 1 number (2))
    labelling in top, middle, and bottom thirds (3 (3 - 5))
    labelling in left, centre, and right thirds (3 (6 - 8))
    two loops, with top, bottom,width, and whether closed at top and bottom (10 (9 - 13 and 14 - 18))
    Then longest lines in top, middle, and bottom thirds (3 (19-21))
    And longest lines in left, middle and right thirds (3 (22-24))
    And number of "lines" in left middle and right thirds (3 (25-27)
    And loop data for top half (5 (28 - 32))
    And loop data for bottom half (5 (33 - 37))"""
          
    #OK, let's find the best bet
    for x in range(10):
        if predictions[x] > predictions[predicted]:
            predicted = x
    
    return predicted                 
 

    
def Process(startpoint,numimages,fileflag):
    """Preprocesses images from MNIST file, extracts features, and saves data.
    
    Takes each 28x28 MNIST image, scales to 1, converts to B&W, and passes
    to the feature extraction routine.
    
    The returned vector for each image is stored in a data file, which can later
    be loaded without reprocessing.
    
    Two parameters are number of images to process, and 0 or 1 for 
    training or test data.
    
    Uncommenting 'showpics = True' will result in the printing of the image and
    data for values of the specified type (where you predicted on thing and got
    another).    """
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    if fileflag == 0:
        images = train_images
        labels = train_labels
    else:
        images = test_images
        labels = test_labels
   
    right=0
    #dmap will store expected (by row) and predicted values, for analysis and tuning
    dmap = np.zeros((10,11),dtype=int)
    for tin in range(startpoint,startpoint+numimages):
        ti = images[tin].copy()
        ti[ti>0.1] = 1.0              #Convert to black and white
        ti[ti<1.0] = 0.0              #Convert to black and white
        fd = FindFeatures(ti)
        predicted = Interpret(fd)
        dmap[labels[tin],predicted] += 1    #increment array at actual, predicted
        showpics = False
        if predicted == labels[tin]:
            right += 1
        else:
            #showpics = True 
            if showpics and labels[tin] == 8 and predicted == 9:               
                DShowImage(ti)
                print(predicted,labels[tin])
                print(fd)
    for x in range(10):
        dmap[x] = dmap[x] * 100 / dmap[x].sum() + 0.5
    print(dmap)
    return right

"""
Here is the start of the main program.

The program will read in the MNIST train and test digit datasets, and process
a subset of 'numimages' of them, starting at 'startpoint'. The third parameter
of Process() is 0 or 1, depending on whether you want to try it against 
the training (0) or test (1) dataset.

Process returns the number of responses it gets right, which is then processed
to give a percentage score."""
startpoint=0
numimages=10000
right = Process(startpoint,numimages,1)
print("I got ",right*100/numimages,"% correct")

