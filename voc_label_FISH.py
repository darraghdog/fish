import xml.etree.ElementTree as ET
import pickle
import os
import json
from os import listdir, getcwd
import PIL 
from PIL import Image 
from os.path import join
os.getcwd()
# os.chdir('C:\\Users\\dhanley2\\Documents\\Personal\\fish\\fish')


sets=[('train-all'), ('test')]
# sets = ['test']

classes = ["ALB", "BET", "DOL", "LAG", "OTHER", "SHARK", "YFT"]

folder_anno_in = 'darknet/FISH/annos'
folder_anno_out = 'darknet/FISH/labels'
folder_img_srce = 'darknet/FISH/JPEGImages'

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

ftype = "YFT"
    
def convert_annojson(ftype, fsrce):
    in_file = open(os.path.join(folder_anno_in,'%s.json'%(ftype))).read()
    tree = json.loads(in_file)
    for ii in range(len(tree)):
        imgjson = tree[ii]
        imgname = str(imgjson['filename'].split('/')[1]).split('.')[0]
        cls_id = str(imgjson['filename'].split('/')[0])
        cls_id = classes.index(cls_id)
        imgsize = PIL.Image.open(folder_img_srce + '/' + fsrce + '/' + imgjson['filename']).size
	if not os.path.exists(os.path.join(folder_anno_out, fsrce, ftype)):
	    os.makedirs(os.path.join(folder_anno_out, fsrce, ftype))

        fo = open(os.path.join(folder_anno_out, fsrce, ftype, '%s.txt'%(imgname)),'w')
        for bb in imgjson['annotations']:
            b = (bb['x'], bb['x']+bb['width'], bb['y'], bb['y']+bb['height'])
            b = convert(imgsize, b)
            fo.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')
            del b
        fo.close()
    
for c in classes:
    print c
    convert_annojson(c, 'train-all')
    
for image_set in sets:
    #if not os.path.exists('batch/labels'):
    #   os.makedirs('batch/labels')
    #if not os.path.exists('batch/ImageSets/Main'):
    #   os.makedirs('batch/ImageSets/Main')
    image_ids = []
    wd = os.getcwd()
    if image_set != 'test':
        for c in classes:
            f = os.listdir(os.path.join(wd, folder_img_srce, image_set, c))
            f = [image_set + '/' + c + '/' + s for s in f]
            image_ids = image_ids + f
    else:
        f = os.listdir(os.path.join(wd, folder_img_srce, image_set))
	f = [image_set + '/' + s for s in f]
	image_ids = f
    list_file = open('%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write(os.path.join(os.getcwd(), folder_img_srce, image_id) + '\n')
        
    list_file.close()
