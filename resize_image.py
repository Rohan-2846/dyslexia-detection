
import os
#import theano
from PIL import Image
from numpy import *

# input image dimensions
img_rows, img_cols = 64, 64

# number of channels
img_channels = 3


path1 = 'E:/final code/Dyslexia disease detection/testing_set/valid/A'    #path of folder of images    
path2 = 'E:/final code/Dyslexia disease detection/testing_set/0'  #path of folder to save images    

listing = os.listdir(path1)
num_samples=size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)  
    img = im.resize((img_rows,img_cols))
    gray = img.convert(mode='RGB')
                #need to do some more processing here          
    gray.save(path2 +'\\' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open('/' + imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images
