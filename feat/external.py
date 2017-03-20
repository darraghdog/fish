# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:29:08 2017

@author: darragh
"""

from logbook import Logger, StreamHandler
import sys
import random
StreamHandler(sys.stdout).push_application()
log = Logger('Logbook')
import xml.etree.ElementTree as ET
import pickle
import urllib
import os
import json
import shutil, csv, time
from os import listdir, getcwd
import PIL 
from PIL import Image 
from os.path import join
import pandas as pd
os.getcwd()
#os.chdir('../')

classes = ["ALB", "BET", "DOL", "LAG", "OTHER", "SHARK", "YFT"]
# folder_anno_in = 'darknet/FISH/annos'
# folder_anno_out = 'darknet/FISH/labels'
# folder_img_srce = 'data/fish'
path = 'data/fish'
refresh_directories = True
# yolo_proba_cutoff = 0.75
# Create the test and valid directory
if refresh_directories:
    log.info('Create directory structure and validation files')
    sub_dirs = os.listdir(os.path.join(path, 'train-all'))
    if '.DS_Store' in sub_dirs: sub_dirs.remove('.DS_Store')
    refresh_directory_structure('crop/train', sub_dirs)
    refresh_directory_structure('crop/valid', sub_dirs)
    refresh_directory_structure('nocrop/train', sub_dirs)
    refresh_directory_structure('nocrop/valid', sub_dirs)
    if os.path.exists(os.path.join(path, 'crop/test/test')):
        shutil.rmtree(os.path.join(path, 'crop/test/test'))
    os.makedirs(os.path.join(path, 'crop/test/test'))
    if os.path.exists(os.path.join(path, 'nocrop/test/test')):
        shutil.rmtree(os.path.join(path, 'nocrop/test/test'))
    os.makedirs(os.path.join(path, 'nocrop/test/test'))

# Create the test and valid directory
if refresh_directories:
    sub_dirs = os.listdir(os.path.join(path, 'train-all'))
    if '.DS_Store' in sub_dirs: sub_dirs.remove('.DS_Store')
    refresh_directory_structure('external/train', sub_dirs)


external = {'ALB' : "https://www.dropbox.com/s/57bqzrla990di61/ALB.tar?dl=1",
            'DOL' : "https://www.dropbox.com/s/ihhwvg5hr42aw9a/DOL.tar?dl=1",
            'LAG' : "https://www.dropbox.com/s/fqumify0gkqg2wi/LAG.tar?dl=1",
            'BET' : "https://www.dropbox.com/s/lngdeoesm40w76f/BET.tar?dl=1",
            'YFT' : "https://www.dropbox.com/s/066wond1ggtohqu/YFT.tar?dl=1",
            'OTHER' : "https://www.dropbox.com/s/qd0npaftd3c8ljf/OTHER.tar?dl=1"}

for c in external:
    print c, external[c]
    urllib.urlretrieve(external[c], 'data/fish/external/%s.tar'%(c))
    tar = tarfile.open('data/fish/external/%s.tar'%(c))
    tar.extractall(path = 'data/fish/external/train/%s'%(c))
    tar.close()