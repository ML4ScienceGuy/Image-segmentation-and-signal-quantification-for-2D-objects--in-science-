#!/usr/bin/env python
# coding: utf-8

# https://stackoverflow.com/questions/70700974/splitting-image-by-whitespace
# https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
# 

# In[1]:


import numpy as np
import pandas as pd
import glob
import os
import csv
import cv2
import sys
import shutil
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
from imutils import perspective
from imutils import contours
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import distance as dist
from skimage import (color, feature, filters, measure, morphology, segmentation, util)
import tifffile as tiff
from statistics import mean
import seaborn as sns
import time

# get the start time
st = time.time()


# In[2]:


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #print("Wrote Image: " + imagePath)

def findBiggestBlob(inputImage):
    # Store a copy of the input image:
    biggestBlob = inputImage.copy()
    # Set initial values for the
    # largest contour:
    largestArea = 0
    largestContourIndex = 0
    
    # Find the contours on the binary image:
    contours, hierarchy = cv2.findContours(inputImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour in the contours list:
    for i, cc in enumerate(contours):
        # Find the area of the contour:
        area = cv2.contourArea(cc)
        # Store the index of the largest contour:
        if area > largestArea:
            largestArea = area
            largestContourIndex = i

    # Once we get the biggest blob, paint it black:
    tempMat = inputImage.copy()
    cv2.drawContours(tempMat, contours, largestContourIndex, (0, 0, 0), -1, 8, hierarchy)
    # Erase smaller blobs:
    biggestBlob = biggestBlob - tempMat

    return biggestBlob


# In[3]:


def make_dir(path,folder):
    newpath= os.path.join(path, folder)
    if os.path.exists(newpath):
        #print("folder exists")
        #print(newpath)
        pass
    else:
        os.mkdir(newpath)
        #print(newpath)


# In[4]:


Path=r"C:\Users\Sorin\Downloads\sorin"
os.chdir(Path)
make_dir(Path,"outputs")
print(Path)


# In[5]:


def make_masks(input_folder,image,output_folder): 
    Image = tiff.imread(input_folder+"\\"+image)
    thresholds = filters.threshold_multiotsu(Image[:,:,0], classes=3)
    red = Image[:, :, 0]
    cells = red > thresholds[0]    
    distance = ndi.distance_transform_edt(cells)
    local_max_coords = feature.peak_local_max(distance, min_distance=7)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    marker = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, marker, mask=cells)

    #print(segmented_cells[0].shape)
    fig, ax = plt.subplots()
    fig.set_size_inches(1024/226.5, 1024/226.5)
    ax.imshow(color.label2rgb(segmented_cells, bg_label=0))
    ax.axis('off')
    plt.show()

    # Save files:
    filepath=output_folder+"\\"+image[:-4]+"_mask.jpg"
    #print(filepath, "==== filepath \n\n")
    fig.savefig(filepath, dpi=300, bbox_inches='tight',pad_inches = 0)


# In[6]:


# Imports

os.chdir(Path)
for folder in glob.glob(Path+"/**")[:-1]: 

    make_dir(folder,"masks")
    make_dir(folder,"splits")
    make_dir(folder,"splits\\cropped")
    for img in glob.glob(folder+"\*.tif"):
        img=img.split("\\")[-1]
        #print(img.split("\\")[-1])
        print(folder)
        print("Image: ", img)
        #CREATE MASKS 
        #print(inputfolder, "is input folder")
        outputfolder=folder+"\\masks"
        make_masks(folder,img,outputfolder)      

        # Read image
        #print (img,"\n",Path + "\\"+img)
        inputImage = cv2.imread(folder+"\\"+img)
        #print("maskImage is:", Path+"\\0b\\masks\\"+img[:-4]+"_mask.jpg")
        maskImage= cv2.imread(folder+"\\masks\\"+img[:-4]+"_mask.jpg")[10:-10, 10:-10]
        #print(maskImage.shape, " is mask image shape")

        # Get image dimensions
        originalImageHeight, originalImageWidth = inputImage.shape[:2]

        # Deep BGR copy:
        colorCopy = inputImage.copy()

        # Convert BGR to grayscale and invert:
        grayscalemask = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)
        grayinv = cv2.bitwise_not(grayscalemask)

        # Threshold via Otsu:
        _, binaryImage = cv2.threshold(grayinv, 240, 255, cv2.THRESH_BINARY_INV)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        binaryImage=cv2.erode(binaryImage,element)

        # Image counter to write pngs to disk:
        imageCounter = 0

        # Segmentation flag to stop the processing loop:
        segmentObjects = True

        while (segmentObjects):

            # Get biggest object on the mask:
            currentBiggest = findBiggestBlob(binaryImage)

            # Use a little bit of morphology to "widen" the mask:
            kernelSize = 3
            opIterations = 2
            morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

            # Mask the original BGR (resized) image:
            blobMask = cv2.bitwise_and(inputImage, inputImage, mask=currentBiggest)

            # Flood-fill at the top left corner:
            fillPosition = (0, 0)
            # Use white color:
            fillColor = (0, 0, 0)
            colorTolerance = (0,0,0)
            cv2.floodFill(blobMask, None, fillPosition, fillColor, colorTolerance, colorTolerance)
            
            #Now we crop. To do that we must find the xy coordinates of the boundaries          
            gray = cv2.cvtColor(blobMask, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # perform edge detection, then perform a dilation + erosion to
            # close gaps in between object edges
            edged = cv2.Canny(gray, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            
            # find contours in the edge map
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #not an image
            cnts = imutils.grab_contours(cnts)
            # sort the contours from left-to-right and initialize the
            # 'pixels per metric' calibration variable
            (cnts, _) = contours.sort_contours(cnts)
            pixelsPerMetric = None
            left=np.min(cnts[0][:,0][:,0])
            right=np.max(cnts[0][:,0][:,0])
            up=np.min(cnts[0][:,0][:,1])            
            down=np.max(cnts[0][:,0][:,1])            
            width=right-left
            height=down-up
            #print(width, height)
            #print(left, right, up, down)

            # Cropping an image, if it is an actual cell and not a cluster of cells (cells are smaller than 180 pixels on any dimension)
            if width> 65 or height> 65 or width<10 or height<10:
                pass
            else:
                global cropfolder
                
                cropfolder=folder+"\\splits\\cropped"
                #print(cropfolder, "is cropfolder")
                cropped_image = blobMask[up:down,left:right] #works
                #dim=(width*10,height*10)
                #resized = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_AREA)
                #print(cropfolder+"\\"+img.split("_")[0]+"_object-"+str(imageCounter), "is cropped image to save")
                writeImage(cropfolder+"\\"+img.split("_")[0]+"_object-"+str(imageCounter), cropped_image)

                #Create transparency for png:
                bg = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                # Threshold to make a mask
                mask = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY)[1]
                # Put mask into alpha channel of Circle
                result = np.dstack((cropped_image, mask))
                writeImage(cropfolder+"\\"+img.split("_")[0]+"_object-"+str(imageCounter), result)
                # Write file to disk:
                #print("split cells go here:",Path+"\\splits\\"+img.split("_")[0]+"_object-"+str(imageCounter))
                writeImage(folder+"\\splits\\"+img.split("_")[0]+"_object-"+str(imageCounter), blobMask)
                imageCounter+=1

            #The rest of the code serves to stop the loop from producing more images than there are objects
            # Subtract current biggest blob to original binary mask:
            binaryImage = binaryImage - currentBiggest

            # Check for stop condition - all pixels in the binary mask should be black:
            whitePixels = cv2.countNonZero(binaryImage)

            # Compare agaisnt a threshold - 10% of resized dimensions:
            whitePixelThreshold = 0.01 * (originalImageHeight* originalImageWidth)
            if (whitePixels < whitePixelThreshold):
                segmentObjects = False



# cv2.imshow("input",inputImage)
# cv2.imshow("mask mask",maskImage)
# cv2.waitKey()

# In[ ]:





# cv2.imshow("copy of Image",colorCopy)
# cv2.imshow("grayscale mask",grayscalemask)
# cv2.imshow("inverted grayscale mask",grayinv)
# cv2.imshow("binary mask",binaryImage)
# cv2.waitKey()

# cv2.imshow("currentbiggest",currentBiggest)
# cv2.imshow("blobmask",blobMask)
# cv2.imshow("gray",gray)
# cv2.imshow("edged",edged)
# #cv2.imshow("cropped image",cropped_image)
# #cv2.imshow("bg",bg)
# #cv2.imshow("mask",mask)
# cv2.waitKey()

# In[ ]:


import glob
count=0

final_data=[["Treatment","File", "Green intensity start", "Green intensity end", "Green most common", "Green most common index",
                     "Red intensity start", "Red intensity end", "Red most common","Red most common index","Green/red most common index",
                     "Red-green most common index", "Index"]]
with open(Path+'//outputs'+'//data.csv', 'w+') as f:        
    f.write('Treatmaent,File,R,G,\n') # saving red and green data
    for folder in glob.glob(Path+"/**")[:-1]:
        cropfolder=folder+"\\splits\\cropped"
        #print(cropfolder)
        os.chdir(cropfolder)
        names=[] #Now we need the names of each file. They contain the experimental conditions and serve as labels for the data
        for infile in glob.glob("*.png"):
            #print(infile.split('.')[0])
            name=infile.split('.')[0]

            # We define an empty data dictionary to keep the pixel data in
            #data_dict={}
            im = Image.open(infile) #relative path to file
            #print(cropfolder,im)

            #load the pixel info 
            pix = im.load() 
            red=[]
            green=[]
            # get a tuple of the x and y dimensions of the image (tuple is (x,y) )
            width, height = im.size 

            #read the details of each pixel and write them to the file 
            for x in range(width): 
                for y in range(height): 
                    count+=1
                    r = pix[x,y][0]
                    red.append(r)
                    g = pix[x,y][1]
                    green.append(g)
                    #print(im, count, "\n",'{0},{1},{2},{3},\n'.format(folder.split("\\")[-1],name,r,g))
                    f.write('{0},{1},{2},{3},\n'.format(folder.split("\\")[-1],name,r,g))
            #data_dict[name]=red,green # storing the pixel data to the dictionary
        #import pprint   
        #pp = pprint.PrettyPrinter()
        #pp.pprint(data_dict)
            #To represent the data as dots
            
            greenranges=[]
            redranges=[]
            img = cv2.imread(infile)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            red_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
            green_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
            blue_hist = cv2.calcHist([img], [2], None, [256], [0, 255])
            greens=[int(x) for x in list(green_hist[:,0])]
            reds=[int(x) for x in list(red_hist[:,0])]
            oranges=[x/y if y!=0 else 0 for x, y in zip(reds,greens)]
            greenmean=mean(greens)
            redmean=mean(reds)
            #print(greenmean,redmean)
            green_hist=green_hist.astype(int) ; red_hist=red_hist.astype(int)
            greensignalrange=[] ; redsignalrange=[] ; green_max_signal=[] ; red_max_signal=[]
            counter=0
            for x,y in zip(green_hist,red_hist):
                counter+=1
                if x>10:        
                    greensignalrange.append([x[0],counter])
                else:
                    pass
                if y>10:        
                    redsignalrange.append([y[0],counter])
                else:
                    pass 
            greensignalrange=sorted(greensignalrange[1:], key=lambda x: x[1])
            greensignalrangef=[greensignalrange[0][1],greensignalrange[-1][1]]
            greenranges.append([name,greensignalrangef,np.max(greens[10:250]),np.argmax(greens[10:250], axis=0)])
            redsignalrange=sorted(redsignalrange[1:], key=lambda x: x[1])
            redsignalrangef=[redsignalrange[0][1],redsignalrange[-1][1]]
            redranges.append([name,redsignalrangef, np.max(reds[10:250]),np.argmax(reds[10:250], axis=0)])


            for x, y in zip(greenranges,redranges):
                final_data.append([folder.split("\\")[-1],x[0],x[1][0],x[1][1],x[2],x[3],y[1][0],y[1][1],y[2],y[3], y[3]-x[3],x[3]/y[3]])

#To save to file the range data
counter=0
with open(Path+'//outputs'+"//statistics on data.csv", 'w', newline='') as file:
    # create the csv writer
    writer = csv.writer(file)
    writer.writerow(final_data[0])
    # write a row to the csv file
    for row in final_data[1:]:
        row.append(counter)
        counter+=1
        writer.writerow(row)




# import glob
# import os
# 
# # We create an empty dictionary that will hold the statistics data
# 
# #We set the folder path
# #os.chdir(cropfolder)
# 
# #Now we need the names of each file. They contain the experimental conditions and serve as labels for the data
# names=[]
# 
# for infile in glob.glob("*.png"):
#     names.append(infile.split('.')[0])
# names    
# 

# In[ ]:


df=pd.DataFrame(pd.read_csv(Path+"\\outputs\statistics on data.csv"))
fig, axes = plt.subplots()
ax=sns.violinplot( x="Treatment", y="Green/red most common index", data=df)
fig = ax.get_figure()
fig.savefig(Path+"//outputs//outputs.png", dpi=300)


# In[ ]:


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', round(elapsed_time/60,2), 'minutes')


# In[ ]:





# In[ ]:





# In[ ]:




