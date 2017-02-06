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

import utils; reload(utils)
from utils import *
import gc
# from __future__ import division, print_function
from theano.sandbox import cuda
from vgg16bn import Vgg16BN

# Set Parameters and check files
input_exists = True
log.info('Set Paramters')
path = "../data/fish/"
batch_size=64

# Read in our VGG pretrained model
log.info('Get VGG')
model = vgg_ft_bn(8)

# Create our VGG model
log.info('Create VGG')
vgg640 = Vgg16BN((360, 640)).model
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
    trn = get_data(path+'train', (360,640))
    val = get_data(path+'valid', (360,640))
    test = get_data(path+'test', (360,640))
    
    # Precompute the output of the convolutional part of VGG
    log.info('Get VGG output')
    conv_val_feat = vgg640.predict(val, batch_size=32, verbose=1)
    conv_trn_feat = vgg640.predict(trn, batch_size=32, verbose=1)
    conv_test_feat = vgg640.predict(test, batch_size=32, verbose=1)
    log.info('Write VGG output')
    save_array(path+'results/conv_val_feat.dat', conv_val_feat)
    save_array(path+'results/conv_trn_feat.dat', conv_trn_feat) 
    save_array(path+'results/conv_test_feat.dat', conv_test_feat)     

    # For memory purposes delete out the original train and validation
    log.info('Clear up memory')
    del trn, val, test
    gc.collect()

conv_val_feat = load_array(path+'results/conv_val_feat.dat')
conv_trn_feat = load_array(path+'results/conv_trn_feat.dat') 
conv_test_feat = load_array(path+'results/conv_test_feat.dat')
    
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

lrg_model = Sequential(get_lrg_layers())
lrg_model.summary()
lrg_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2, 
             validation_data=(conv_val_feat, val_labels))
lrg_model.optimizer.lr=1e-5
lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6,
             validation_data=(conv_val_feat, val_labels))

# Evaluate the model
log.info('Evaluate')
lrg_model.evaluate(conv_val_feat, val_labels)

# Make our prediction on the lrg_model layer
log.info('Output Prediction')
preds = lrg_model.predict(conv_test_feat, batch_size=batch_size) # or try 32 batch_size
subm = do_clip(preds,0.99)
subm_name = path+'results/subm_bb_conv_lrg0202A.csv.gz'
pred_name = path+'results/pred_bb_conv_lrg0202A.csv.gz'

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', raw_test_filenames)
submission.to_csv(subm_name, index=False, compression='gzip')
subm1 = pd.DataFrame(preds, columns=classes)
subm1.insert(0, 'image', raw_test_filenames)
subm1.to_csv(pred_name, index=False, compression='gzip')




log.info('Done - files @ ' + subm_name)





