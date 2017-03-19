# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:45:32 2017

@author: darragh
"""

import os, math, glob
import ujson as json
from PIL import Image
import numpy as np
from shutil import make_archive, copy2



TRAIN_DIR = 'data/fish/train-all/'
TEST_DIR = 'data/fish/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# Set our directories
path = "/home/darragh/Dropbox/fish"
os.chdir(path)
anno_path = "darknet/FISH"

#Create Annotations directory
PATH = 'rcnn/annosxml'
    
def refresh_dir(PATH):
    if not os.path.exists(os.path.join(anno_path, PATH)):
        os.mkdir(os.path.join(anno_path, PATH))
    files = glob.glob(os.path.join(anno_path, PATH,'*'))
    for f in files:
        os.remove(f)
refresh_dir('rcnn')
refresh_dir('rcnn/FISH')
refresh_dir('rcnn/FISH/Annotations')
refresh_dir('rcnn/FISH/ImageSets')
refresh_dir('rcnn/FISH/ImageSets/Main')
refresh_dir('rcnn/FISH/JPEGImages')


# Start annotating the images with fish
crop_classes=FISH_CLASSES[:]
crop_classes.remove('NoF')
crop_classes

for c in crop_classes:
    j = json.load(open(os.path.join(anno_path, 'annos/{}.json'.format(c)), 'r'))
    for l in j: 
        filename = l["filename"]
        head, tail = os.path.split(filename)
        basename, file_extension = os.path.splitext(tail) 
        if len(l["annotations"]) == 0:
            print(filename)
            print("no bbox")
        else:
            f = open(os.path.join(anno_path, 'rcnn/FISH/Annotations/' + basename + '.xml'),'w') 
            line = "<annotation>" + '\n'
            f.write(line)
            line = '\t<folder>' + 'FISH' + '</folder>' + '\n'
            f.write(line)
            line = '\t<filename>' + tail + '</filename>' + '\n'
            f.write(line)
            line = '\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
            f.write(line)
            im=Image.open(TRAIN_DIR+ c + '/' + tail)
            (width, height) = im.size
            line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + \
            str(height) + '</height>\n\t\t<depth>3</depth>\n\t</size>'
            f.write(line)
            line = '\n\t<segmented>0</segmented>'
            f.write(line)
            for a in l["annotations"]:
                line = '\n\t<object>'
                line += '\n\t\t<name>' + 'fish' + '</name>\n\t\t<pose>Unspecified</pose>' # a["class"].lower()
                #line += '\n\t\t<name>fish</name>\n\t\t<pose>Unspecified</pose>'
                line += '\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>'
                xmin = (a["x"])
                line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
                ymin = (a["y"])
                line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
                width = (a["width"])
                height = (a["height"])
                xmax = xmin + width
                ymax = ymin + height
                line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
                line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
                line += '\n\t\t</bndbox>'
                line += '\n\t</object>'     
                f.write(line)
            line = '</annotation>'
            f.write(line)
            f.close()

  
#write ImageSets/Main
imgs = []
for fish in crop_classes:
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    imgs_fish = [os.path.splitext(im)[0] for im in os.listdir(fish_dir)]
    imgs.extend(imgs_fish)
index = np.random.permutation(len(imgs))
imgs = [imgs[i] for i in index]
num_train = math.ceil(len(imgs)*0.7)
with open(os.path.join(anno_path, 'rcnn/FISH/ImageSets/Main', 'train.txt'),'w') as f:
    train = sorted(imgs[:int(num_train)])
    for im in train:
        f.write(im + '\n')
with open(os.path.join(anno_path, 'rcnn/FISH/ImageSets/Main', 'val.txt'),'w') as f:
    val = sorted(imgs[int(num_train):])
    for im in val:
        f.write(im + '\n')
with open(os.path.join(anno_path, 'rcnn/FISH/ImageSets/Main', 'trainval.txt'),'w') as f:
    trainval = sorted(imgs)
    for im in trainval:
        f.write(im + '\n')
#del img_00568 and img_07439
imgs_fish = []
for fish in FISH_CLASSES:
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    imgs_fish += [os.path.splitext(im)[0] for im in os.listdir(fish_dir)]
train_fish = [im+'  1' if im in imgs_fish else im+' -1' for im in train]
val_fish = [im+'  1' if im in imgs_fish else im+' -1' for im in val]
trainval_fish = [im+'  1' if im in imgs_fish else im+' -1' for im in trainval]
with open(os.path.join(anno_path, 'rcnn/FISH/ImageSets/Main', 'FISH_train.txt'),'w') as f:
    for im in train_fish:
        f.write(im + '\n')
with open(os.path.join(anno_path, 'rcnn/FISH/ImageSets/Main', 'FISH_val.txt'),'w') as f:
    for im in val_fish:
        f.write(im + '\n')
with open(os.path.join(anno_path, 'rcnn/FISH/ImageSets/Main', 'FISH_trainval.txt'),'w') as f:
    for im in trainval_fish:
        f.write(im + '\n')
            
# Write the same for the test set
fish_dir = TEST_DIR
imgs_test = [os.path.splitext(im)[0] for im in os.listdir(fish_dir)]
with open(os.path.join(anno_path, 'rcnn/FISH/ImageSets/Main', 'test.txt'),'w') as f:
    for im in sorted(imgs_test):
        f.write(im + '\n')
        
# Move all the images to JPEGImages file
for fish in FISH_CLASSES:
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    imgs = os.listdir(fish_dir)
    for img_ in imgs:
        copy2(os.path.join(fish_dir, img_), 
              os.path.join(anno_path, 'rcnn/FISH/JPEGImages', img_))
fish_dir = TEST_DIR
imgs = os.listdir(TEST_DIR)
for img_ in imgs:
    copy2(os.path.join(fish_dir, img_), 
          os.path.join(anno_path, 'rcnn/FISH/JPEGImages', img_))      

# Zip the folder for rcnn to move to RCNN server/folder
make_archive(
  'rcnn_PASCAL_FISH', 
  'zip',           # the archive format - or tar, bztar, gztar 
  root_dir=os.path.join(path, anno_path, 'rcnn/'),   # root for archive - current working dir if None
  base_dir=os.path.join(path, anno_path, 'rcnn/'))   # start archiving from here - cwd if None too