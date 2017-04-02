
import os, random, glob, pickle, collections, math
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
import gc
import glob
import shutil, csv, time

import utils; reload(utils)
from utils import *

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('tf')


# In[2]:

TRAIN_DIR = '../data/fish/train-all/'
TEST_DIR =  '../data/fish/test/' 
CHECKPOINT_DIR = './checkpoints/checkpoint04B/'
LOG_DIR = './logs'
FISH_CLASSES = ['NoF', 'ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
CONF_THRESH = 0.8
ROWS = 224
COLS = 224
BATCHSIZE = 32 # 256 #64
LEARNINGRATE = 1e-4
BG_THRESH_HI = 0.3
BG_THRESH_LO = 0.1
bags = 5
learn_round = 2
p=16
full = True



def load_img(path, bbox, target_size=None):
    img = Image.open(path)
    imsize = Image.open(path).size
    height, width = bbox[2]-bbox[0], bbox[3]-bbox[1]
    length = max(height, width)    
    # Make it square
    dim = [width, height]
    for i in range(2):
        offset = length - dim[0+i]
        if bbox[0+i]+length+(offset/2) > imsize[0+i]:
            bbox[0+i] = bbox[2+i] - length + (offset/2)
            bbox[2+i] = bbox[2+i] + (offset/2)
        else:
            bbox[2+i] = bbox[0+i] + length
        bbox[0+i] -= length*0.05
        bbox[2+i] += length*0.05
        
    img = img.convert('RGB')
    cropped = img.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
    if target_size:
        cropped = cropped.resize((target_size[1], target_size[0]))
    if height < width:
        cropped = cropped.rotate(-90)
    return cropped

def preprocess_input(x):
    #resnet50 image preprocessing
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    x[:, :, 0] -= 105
    x[:, :, 1] -= 115
    x[:, :, 2] -= 123
    return x


'''
'''
train_datagen = ImageDataGenerator(
    rotation_range=180,
    shear_range=0.2,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)


# Load up YOLO bounding boxes for each class
import glob
# all_files = glob.glob(os.path.join('../yolo_coords', "*.txt"))
# allFiles = [f for f in all_files if 'FISH' in f]
all_files = glob.glob(os.path.join('../yolo_coords', "*.txt"))
allFiles = [f for f in all_files if 'FISH544.' in f]
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=None, sep = " ", names = ['fname', 'proba', 'x0', 'y0', 'x1', 'y1'])
    df['class'] = file_.split('_')[-1].split('.')[0]
    list_.append(df)
yolo_frame = pd.concat(list_)
# Cut off the predictions on a probabilty
yolo_frame['proba_max'] = df.groupby('fname')['proba'].transform('max')
yolo_frame = yolo_frame[yolo_frame['proba_max']<0.7]
yolo_frame = yolo_frame[yolo_frame['proba_max']>0.5]
# Sort the predictions on the area 
yolo_frame['area'] = (yolo_frame['x1']-yolo_frame['x0']) * (yolo_frame['y1']-yolo_frame['y0'])
yolo_frame = yolo_frame.sort(['fname','area'], ascending=[1, 0]).reset_index(drop=True)


# In[ ]:

file_name = 'GTbbox_test544_df.pickle'
if False: #os.path.exists('../data/'+file_name):
    print ('Loading from file '+file_name)
    GTbbox_test_df = pd.read_pickle('../data/'+file_name)
else:
    print ('Generating file '+file_name)       
    GTbbox_test_df = pd.DataFrame(columns=['image_folder', 'image_file','crop_index','crop_class','xmin','ymin','xmax','ymax'])  
    iddict = {}
    for c in ['test']:
        print(c)
        for l in range(yolo_frame.shape[0]): 
            image_file, proba, xmin, ymin, xmax, ymax, fish_class, area = yolo_frame.iloc[l].values.tolist()    
            if image_file in iddict:
                iddict[image_file] += 1
            else:
                iddict[image_file] = 0
            image = Image.open(TEST_DIR+c+'/'+image_file+'.jpg')
            width_image, height_image = image.size
            width = xmax - xmin
            height = ymax - ymin
            delta_width = p/(COLS-2*p)*width
            delta_height = p/(ROWS-2*p)*height
            xmin_expand = xmin-delta_width
            ymin_expand = ymin-delta_height
            xmax_expand = xmin+width+delta_width
            ymax_expand = ymin+height+delta_height
            assert max(xmin_expand,0)<min(xmax_expand,width_image)
            assert max(ymin_expand,0)<min(ymax_expand,height_image)
            GTbbox_test_df.loc[len(GTbbox_test_df)] = [c, image_file+'.jpg', iddict[image_file],fish_class,max(xmin_expand,0),max(ymin_expand,0),min(xmax_expand,width_image),min(ymax_expand,height_image)]                    
    GTbbox_test_df = GTbbox_test_df.sort(['image_file','crop_index']).reset_index(drop=True)
    GTbbox_test_df.to_pickle('../data/'+file_name)


gc.collect()



def test_generator(df, datagen = None, batch_size = BATCHSIZE):
    n = df.shape[0]
    batch_index = 0
    while 1:
        current_index = batch_index * batch_size
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1    
        else:
            current_batch_size = n - current_index
            batch_index = 0        
        batch_df = df[current_index:current_index+current_batch_size]
        batch_x = np.zeros((batch_df.shape[0], ROWS, COLS, 3), dtype=K.floatx())
        i = 0
        for index,row in batch_df.iterrows():
            image_file = row['image_file']
            bbox = [row['xmin'],row['ymin'],row['xmax'],row['ymax']]
            cropped = load_img(TEST_DIR+image_file,bbox,target_size=(ROWS,COLS))
            x = np.asarray(cropped, dtype=K.floatx())
            if datagen is not None: x = datagen.random_transform(x)            
            x = preprocess_input(x)
            batch_x[i] = x
            i += 1
        if batch_index%50 == 0: print(batch_index)
        yield(batch_x)


# In[ ]:

import glob
files = glob.glob(CHECKPOINT_DIR+'*')
val_losses = [float(f.split('-')[-1][:-5]) for f in files]
min_id = np.array(val_losses).argsort()[:bags].tolist()


# In[ ]:

# Loop the the lowest val losses and get a prediction for each
test_preds_ls = []
for index in min_id:
    index = val_losses.index(min(val_losses))
    print('Loading model from checkpoints file ' + files[index])
    test_model = load_model(files[index])
    test_model_name = files[index].split('/')[-2][-1:]+'_'+files[index].split('/')[-1]
    test_preds_ls.append(test_model.predict_generator(test_generator(df=GTbbox_test_df, datagen = train_datagen), 
                                         val_samples=GTbbox_test_df.shape[0])) 
    del test_model
    gc.collect()



test_preds = sum(test_preds_ls)/len(test_preds_ls)


# In[ ]:

GTbbox_test_df[:3]


# In[ ]:

columns = ['NoF', 'ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
yolo_pred_df = pd.DataFrame(test_preds, columns=columns)



yolo_pred_df['image_file'] = GTbbox_test_df.image_file
yolo_pred_df['crop_index'] = GTbbox_test_df.crop_index
yolo_pred_df[yolo_pred_df['crop_index']<2].shape


# In[ ]:

yolo_pred_df = yolo_pred_df.groupby(['image_file'], as_index=False).mean().reset_index(drop=True)
yolo_pred_df.drop('crop_index', axis=1, inplace=True)


# In[ ]:

timestr = time.strftime("%Y%m%d")
if full:
    subm_name = '../sub/subm_full_loyolo_resnet_5B.csv'
else:
    subm_name = '../sub/subm_part_loyolo_resnet_5B.csv'



yolo_pred_df.to_csv(subm_name, index=False)#, compression='gzip')
