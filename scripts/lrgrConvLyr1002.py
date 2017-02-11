# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 09:31:22 2017

@author: dhanley2
"""

# Read in Libraries
from __future__ import division, print_function
from logbook import Logger, StreamHandler
import sys
StreamHandler(sys.stdout).push_application()
log = Logger('Logbook')
import shutil, csv, time
timestr = time.strftime("%Y%m%d")

import utils; reload(utils)
from utils import *
import gc
# from __future__ import division, print_function
from theano.sandbox import cuda
from vgg16bn import Vgg16BN
from sklearn import metrics

def accuracyfunc(y_act, y_pred):
    return metrics.accuracy_score(np.argmax(y_act, axis=1), np.argmax(y_pred, axis=1))
    
def refresh_directory_structure(name, sub_dirs):
    gdir = os.path.join(path, name)
    if os.path.exists(gdir):
        shutil.rmtree(gdir)
    os.makedirs(gdir)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(gdir, sub_dir))

# Set Parameters and check files
refresh_directories = False
input_exists = True
full = True
log.info('Set Paramters')
path = "../data/fish/"
batch_size=64
clip = 0.99
bags = 6

# Create the test and valid directory
if refresh_directories:
    log.info('Create directory structure and validation files')
    sub_dirs = os.listdir(os.path.join(path, 'train-all'))
    if '.DS_Store' in sub_dirs: sub_dirs.remove('.DS_Store')
    refresh_directory_structure('train', sub_dirs)
    refresh_directory_structure('valid', sub_dirs)
    for c,row in enumerate(csv.DictReader(open('../image_validation_set.csv'))):
        value = 'valid' if row['Validation'] == '1' else 'train'
        name_from = os.path.join(path, 'train-all', row['SubDirectory'], row['file_name'])
        name_to   = os.path.join(path, value, row['SubDirectory'], row['file_name'])
        shutil.copyfile(name_from, name_to)
        
# Read in our VGG pretrained model
log.info('Get VGG')
model = vgg_ft_bn(8)

# Create our VGG model
log.info('Create VGG')
#vgg640 = Vgg16BN((360, 640)).model
vgg640 = Vgg16BN((720, 1280)).model
vgg640.pop()
vgg640.input_shape, vgg640.output_shape
vgg640.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

# get labels
(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)

# Read in filenames
log.info('Read filenames')
raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]


log.info('Read in data')
if not input_exists:

    batches = get_batches(path+'train', batch_size=batch_size)
    val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)
    (val_classes, trn_classes, val_labels, trn_labels, 
        val_filenames, filenames, test_filenames) = get_classes(path)
    
    # Fetch our large images 
    log.info('Fetch images')
    val = get_data(path+'valid', (720, 1280))
    conv_val_feat = vgg640.predict(val, batch_size=32, verbose=1)
    save_array(path+'results/conv_val_big_feat.dat', conv_val_feat)
    del val, conv_val_feat
    gc.collect()
    
    trn = get_data(path+'train', (720, 1280))
    conv_trn_feat = vgg640.predict(trn, batch_size=32, verbose=1)
    save_array(path+'results/conv_trn_big_feat.dat', conv_trn_feat) 
    del trn, conv_trn_feat
    gc.collect()
    
    test = get_data(path+'test', (720, 1280))
    conv_test_feat = vgg640.predict(test, batch_size=32, verbose=1)
    save_array(path+'results/conv_test_big_feat.dat', conv_test_feat) 
    del test, conv_test_feat 
    gc.collect()    

    # For memory purposes delete out the original train and validation
    log.info('Clear up memory')
    gc.collect() 

conv_val_feat = load_array(path+'results/conv_val_big_feat.dat')
conv_trn_feat = load_array(path+'results/conv_trn_big_feat.dat') 
conv_test_feat = load_array(path+'results/conv_test_big_feat.dat')

if full:
    conv_trn_feat = np.concatenate([conv_trn_feat, conv_val_feat])
    trn_labels = np.concatenate([trn_labels, val_labels]) 
    
# Our Convolutional Net Architecture
log.info('Create and fit CNN')
def get_lrg_layers():
    return [
        BatchNormalization(axis=1, input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D((1,2)),
        Convolution2D(8,3,3, border_mode='same'),
        Dropout(p),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]

# Set up the fully convolutional net (FCN); 
conv_layers,_ = split_at(vgg640, Convolution2D)
nf=128; p=0. # No dropout

lrg_model = []
predsls = []
pvalsls = []

for i in range(bags):

    log.info('Train round' + str(i))
    lrg_model.append(Sequential(get_lrg_layers()))
    if i == 0:
        lrg_model[i].summary()
    lrg_model[i].compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    lrg_model[i].fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2,
                 validation_data=(conv_val_feat, val_labels))
    lrg_model[i].optimizer.lr=1e-7
    lrg_model[i].fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6,
                 validation_data=(conv_val_feat, val_labels))

    # Make our prediction on the lrg_model layer
    log.info('Output Prediction')
    predsls.append(lrg_model[i].predict(conv_test_feat, batch_size=batch_size)) # or try 32 batch_size
    pvalsls.append(lrg_model[i].predict(conv_val_feat, batch_size=batch_size))
    val_score = "%.3f" % metrics.log_loss(val_labels, sum(pvalsls)/len(pvalsls))
    acc_score = "%.3f" % accuracyfunc(val_labels, do_clip(sum(pvalsls)/len(pvalsls), clip))
    log.info('Bagged Validation Logloss ' + str(val_score))
    log.info('Bagged Validation Accuracy ' + str(acc_score))
    # 10 bagged : 0.131

# metrics.log_loss(val_labels, do_clip(sum(pvalsls)/len(pvalsls), .9999))
preds = sum(predsls)/len(predsls)
subm = do_clip(preds, clip)

if full:
    subm_name = path+'results/subm_full_conv_' + timestr + '.csv.gz'
else:
    subm_name = path+'results/subm_part_conv_' + timestr + '.csv.gz'

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', raw_test_filenames)
submission.to_csv(subm_name, index=False, compression='gzip')
log.info('Done - files @ ' + subm_name)

# Bag 6 Original scores 
#[2017-02-09 22:40:05.864336] INFO: Logbook: Bagged Validation Logloss 1.046
#[2017-02-09 22:40:05.864498] INFO: Logbook: Bagged Validation Accuracy 0.706

