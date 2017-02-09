# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:37:48 2017

@author: darragh
"""
import multiprocessing
import zipfile
import os
import io
from PIL import Image
current_dir = '/home/darragh/Dropbox/fish/data'
current_dir = 'C:\\Users\\dhanley2\\Documents\\Personal\\fish\\fish\\data'
os.chdir(current_dir)
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn import cluster

def hamdist(hash_set):
    diffs = 0
    for ch1, ch2 in zip(hash_set[0], hash_set[1]):
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
subdir = os.listdir(os.path.join(current_dir, parentdir))
img_id_hash = []
counter = 1
hash_size = 16
pool = multiprocessing.Pool(4)

for direc in subdir: 
    try:
        names = os.listdir(parentdir + "/" + direc)
    except:
        continue
    print counter, direc
    for name in names:
        imgdata = Image.open(os.path.join(parentdir, direc, name)).convert("L")
        img_hash = dhash(imgdata, hash_size)
        img_id_hash.append([parentdir, direc, name, img_hash])
        counter+=1

parentdir = "test"
names = os.listdir(os.path.join(current_dir, parentdir))
for name in names:
    imgdata = Image.open(os.path.join(parentdir, name)).convert("L")
    img_hash = dhash(imgdata, hash_size)
    img_id_hash.append([parentdir, 'test', name, img_hash])
    counter+=1

df = pd.DataFrame(img_id_hash,columns=['ParDirectory' , 'SubDirectory', 'file_name', 'image_hash'])

# Create the distance matrix 
distances = np.zeros((df.shape[0], df.shape[0]))
for i, row in df.iterrows():
    if i % 20 == 0:
        print i
    all_hashes = [(row['image_hash'], f) for f in df.image_hash.tolist()]
    dists = map(hamdist, all_hashes)
    distances[i, :] = dists

# Get a histogram of the distances
plt.hist(distances.flatten(), bins=50)
plt.title('Histogram of distance matrix')

# Cluster the images
cls = cluster.KMeans(n_clusters=df.shape[0]/5)
y = cls.fit_predict(distances)

# show the size of the first 50 clusters
print(pd.Series(y).value_counts()[:50])

# Lets look at the first 5 clusters
_, ax = plt.subplots(15, 4, figsize=(10, 30))
ax = ax.flatten()
count = 0

for c in range(20):
    for i, row in df[y==c].iterrows():
        if count  == len(ax) : break
        if row['ParDirectory'] == 'test' :
            imgdata = Image.open(os.path.join(row['ParDirectory'], row['file_name']))
        else:
            imgdata = Image.open(os.path.join(row['ParDirectory'], row['SubDirectory'], row['file_name']))
        axis = ax[count]
        axis.set_title('Cluster ' + str(c) + ' ' + row['SubDirectory'], fontsize=10)
        axis.imshow(np.asarray(imgdata), interpolation='nearest', aspect='auto')        
        axis.axis('off')
        count += 1
df['cluster'] = y

df.to_csv('image_hash_clusters.csv', index=False)

