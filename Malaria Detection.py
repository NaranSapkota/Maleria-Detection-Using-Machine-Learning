#Malaria detection in ML

#Importing the libarary 
import cv2,os
import numpy as np
import csv
import glob

#Importing the dataset
label="Uninfected"
dirList=glob.glob("cell_images/"+ label +"*/.png")
file=open("D:/dataset.csv","a")

#Accessing files
for img_path in dirList:
    Image=cv2.imread("img_path")
    Image=cv2.GaussainBlur(Image,(5*5),2)  #image smooth 
    Image_Grey=cv2.cvtColor(Image,cv2.COLOR_BGR2GREY) #Converting image into greyscale
    
    ret,thre=cv2.Threshold(Image_Grey,127,254,0)  # Applying threhold value to every pixel
    contours=cv2.findContours(thre,1,2)
    
    file.write(label)
    file.write(",")
    
    for i in range(5):
         try:
             area=cv2.contourArea(contours[i])
             file.write(str(area))
             
         except:
            file.write("0")
    
         file.write(",")
        
    file.write("\n")
    

