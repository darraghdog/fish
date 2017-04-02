import os
import io
from PIL import Image
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import multiprocessing
from sklearn import cluster
import random
random.seed(100);

# Set working directory
os.chdir('../data')

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

counter = 1
hash_size = 16
duplicate_images = False
duplicates = True

img_id_hash = []
parent_dir = "fish/test/test"
names = os.listdir(parent_dir)
for name in names:
    imgdata = Image.open(os.path.join(parent_dir, name)).convert("L")
    img_hash = dhash(imgdata, hash_size)
    img_id_hash.append([parent_dir, name, img_hash])
    counter+=1

df = pd.DataFrame(img_id_hash,columns=['ParDirectory' , 'file_name', 'image_hash'])
df.head()

# Create the distance matrix for the distances in images
pool = multiprocessing.Pool(8)
distances = np.zeros((df.shape[0], df.shape[0]))
for i, row in df.iterrows():
    #if i % 50 == 0: print i
    all_hashes = [(row['image_hash'], f) for f in df.image_hash.tolist()]
    dists = pool.map(hamdist, all_hashes)
    distances[i, :] = dists
    
# Lets look at the first 5 clusters
if duplicate_images:
    _, ax = plt.subplots(112, 4, figsize=(24, 560))
    ax = ax.flatten()
    counter = 0
    for c in range(200):
        if counter  == len(ax): 
            break
        for i in range(1000):
            for j in range(i,1000):
                if (distances[i,j]>10) and (distances[i,j]<15):
                    if i!=j:
                        imgdata = Image.open(os.path.join('fish/test/test', names[i]))
                        axis = ax[counter]
                        axis.imshow(np.asarray(imgdata), interpolation='nearest', aspect='auto')        
                        axis.axis('off')
                        counter += 1
                        imgdata = Image.open(os.path.join('fish/test/test', names[j]))
                        axis = ax[counter]
                        axis.imshow(np.asarray(imgdata), interpolation='nearest', aspect='auto')        
                        axis.axis('off')
                        counter += 1
if duplicates:
    imgls = []
    for i in range(1000):
        for j in range(i,1000):
            if (distances[i,j]<25): # (distances[i,j]>10) and 
                if i!=j:
                    #print(counter, names[i], names[j])
                    check = 0 
                    for subls in imgls:
                        if any(x in subls for x in [names[i], names[j]]):
                            subls += [names[i], names[j]]
                            check = 1
                        if check ==1 :continue
                    if check == 0: imgls.append([names[i], names[j]])
for i in range(len(imgls)):
    imgls[i] = list(set(imgls[i]))
    
# Open the file to write to
fo = open("../test_fish_same_test.csv", 'w')
fo.write('image,group\n')
for subls in imgls:
    for im in subls:
        #print (im, ' '.join(subls))
        fo.write('%s,%s\n' % (im, ' '.join(subls)))
fo.close()