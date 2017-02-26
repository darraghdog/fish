from logbook import Logger, StreamHandler
import sys
StreamHandler(sys.stdout).push_application()
log = Logger('Logbook')
import xml.etree.ElementTree as ET
import pickle
import os
import json
import shutil, csv, time
from os import listdir, getcwd
import PIL 
from PIL import Image 
from os.path import join
import pandas as pd
os.getcwd()
# os.chdir('C:\\Users\\dhanley2\\Documents\\Personal\\fish\\fish')
os.chdir('/home/darragh/Dropbox/fish/')

classes = ["ALB", "BET", "DOL", "LAG", "OTHER", "SHARK", "YFT"]
folder_anno_in = 'darknet/FISH/annos'
folder_anno_out = 'darknet/FISH/labels'
folder_img_srce = 'data/fish'
path = 'data/fish'
refresh_directories = True
yolo_proba_cutoff = 0.4


def refresh_directory_structure(name, sub_dirs):
    gdir = os.path.join(path, name)
    if os.path.exists(gdir):
        shutil.rmtree(gdir)
    os.makedirs(gdir)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(gdir, sub_dir))

# Create the test and valid directory
if refresh_directories:
    log.info('Create directory structure and validation files')
    sub_dirs = os.listdir(os.path.join(path, 'train-all'))
    if '.DS_Store' in sub_dirs: sub_dirs.remove('.DS_Store')
    refresh_directory_structure('crop/train', sub_dirs)
    refresh_directory_structure('crop/valid', sub_dirs)
    if os.path.exists(os.path.join(path, 'crop/test')):
        shutil.rmtree(os.path.join(path, 'crop/test'))
    os.makedirs(os.path.join(path, 'crop/test'))

# Read in the validation set
df_valid = pd.read_csv('image_validation_set.csv')
df_valid['image_folder'] = df_valid['SubDirectory'] + '/' + df_valid['file_name']

# Function to offset boundary box correctly
def bbox_offset(x, y, offset, size, padding = 0, cut = 0):
    if cut == 1: x = x - w/3
    if cut == 2: y = y - h/3
    if cut == 3: x = x + w/3
    if cut == 4: y = y + h/3
    x, y, offset = x - padding/2, y - padding/2, offset + padding
    if x < 0.0:
        x = 0.0
    elif x + offset > size[0]:
        x = size[0] - offset
    if y < 0.0:
        y = 0.0
    elif y + offset > size[1]:
        y = size[1] - offset
    return x, y, offset+x, offset+y
    
# Make the train and valid images
for ftype in classes:
    print ftype
    in_file = open(os.path.join(folder_anno_in,'%s.json'%(ftype))).read()
    tree = json.loads(in_file)
    for ii in range(len(tree)):
        imgjson = tree[ii]
        validation = df_valid[df_valid['image_folder'] == str(imgjson['filename'])].values
        subdir = validation[0][1]
        topdir = ['train', 'valid'][validation[0][5]]
        fname = validation[0][2].split('.')[0]
        img = PIL.Image.open(os.path.join(folder_img_srce, 'train-all', validation[0][6]))
        for a in range(len(imgjson['annotations'])):
            imgano = imgjson['annotations'][a]
            x, y, w0, h0 = imgano['x'], imgano['y'], imgano['width'], imgano['height']
            # make it a box
            w, h = max(h0, w0), max(h0, w0)
            # centre it
            x, y = x - (w-w0)/2, y - (h-h0)/2 
            img.crop((x, y, h+x, w+y))
            # Avoid borders 
            for pad in [0, 200, 400]:
                cut = 0
                fo = '%s_%s_%s_cut%s.jpg'%(fname, a, pad, cut)
                img.crop(bbox_offset(x, y, h, img.size, pad, cut)).save(os.path.join(folder_img_srce, 'crop', topdir, subdir, fo))
            # Avoid borders 
            for cut in [1,2,3,4]:
                pad = 0
                fo = '%s_%s_%s_cut%s.jpg'%(fname, a, pad, cut)
                img.crop(bbox_offset(x, y, h, img.size, pad, cut)).save(os.path.join(folder_img_srce, 'crop', topdir, subdir, fo))

# Now read in the yolo bindings
yolo_files = os.listdir('yolo_coords')
list_ = []
colnames = ['fname', 'proba', 'x', 'y', 'w', 'h']
for file_ in yolo_files :
    df = pd.read_csv(os.path.join('yolo_coords', file_),index_col=None, header=None, sep = " ", names = colnames)
    list_.append(df)
yolodf = pd.concat(list_, axis = 0, ignore_index=True)
# Get the max value by image
yolodf = yolodf.iloc[yolodf.groupby(['fname']).apply(lambda x: x['proba'].idxmax())].reset_index(drop=True)
yolodf = yolodf[yolodf['proba'] > 0.6]#yolo_proba_cutoff]

# Make the test images
for ii in range(yolodf.shape[0]):
    yoloc = yolodf.iloc[ii].values
    fname = yoloc[0]
    img = PIL.Image.open(os.path.join(folder_img_srce, 'test', fname)+'.jpg')
    x, y, w0, h0 = yoloc[2], yoloc[3], yoloc[4] - yoloc[2], yoloc[5] - yoloc[2]        # make it a box
    w, h = max(h0, w0), max(h0, w0)
    # centre it
    x, y = x - (w-w0)/2, y - (h-h0)/2 
    pad = 200
    fo = '%s_%s.jpg'%(fname, pad)
    img.crop(bbox_offset(x, y, h, img.size, pad)).save(os.path.join(folder_img_srce, 'crop', 'test', fo))

