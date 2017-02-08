# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:37:48 2017

@author: darragh
"""

import zipfile
import os
import io
from PIL import Image
#os.chdir('/home/run2/avito')
current_dir = '/home/darragh/Dropbox/fish/data'
os.chdir(current_dir)
import pandas as pd
import datetime

def hamdist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    pixels = list(image.getdata())
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    
    return ''.join(hex_string)
    
parentdir = "train-all"
subdir = os.listdir(current_dir + "/" + parentdir)
img_id_hash = []
counter = 1

for direc in subdir: # [0,1,2,3,4,5,6,7,8,9]:  
    try:
        names = os.listdir(parentdir + "/" + direc)
    except:
        continue
    print counter, direc
    for name in names:
        imgdata = Image.open(parentdir + '/' + direc + '/' + name).convert("L")
        img_hash = dhash(imgdata, 16)
        img_id_hash.append([parentdir, direc, name, img_hash])
        counter+=1

parentdir = "test"
names = os.listdir(current_dir + "/" + parentdir)
for name in names:
    imgdata = Image.open(parentdir + '/' + name).convert("L")
    img_hash = dhash(imgdata, 16)
    img_id_hash.append([parentdir, 'test', name, img_hash])
    counter+=1

df = pd.DataFrame(img_id_hash,columns=['ParDirectory' , 'SubDirectory', 'file_name', 'image_hash'])
df = df.sort(['image_hash'], ascending=[1]).reset_index(drop=True)
for i in range(1, df.shape[0]):
    df.loc[i, 'distance']  = hamdist(df.loc[i, 'image_hash'], df.loc[i-1, 'image_hash'])


df.to_csv('image_hash_' + parentdir + '.csv', index=False)

