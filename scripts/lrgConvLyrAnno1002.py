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
import ujson as json
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
bags = 1
load_size = (360, 640)

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
vgg640 = Vgg16BN(load_size).model
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

# Read in the boxes
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
bb_json = {}
for c in anno_classes:
    j = json.load(open(os.path.join(path, 'box/{}_labels.json'.format(c)), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]

empty_bbox = {'height': 0., 'width': 0., 'x': 100., 'y': 100.}
for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox

# Finally, we convert the dictionary into an array, and convert the coordinates to our resized 224x224 images.
bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (load_size[1] / size[0])
    conv_y = (load_size[0] / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    bbout = []
    bbout.append(bb[2]+.5*bb[1])
    bbout.append(bb[3]+.5*bb[0])
    return bbout


trn_sizes = [PIL.Image.open(path+'train/'+f).size for f in filenames]
val_sizes = [PIL.Image.open(path+'valid/'+f).size for f in val_filenames]
tst_sizes = [PIL.Image.open(path+'test/'+f).size for f in test_filenames]

trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, trn_sizes)], 
                   ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s) 
                   for f,s in zip(raw_val_filenames, val_sizes)]).astype(np.float32)
                       

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(i):
    bb = val_bbox[i]
    img  = np.rollaxis(val[i], 0, 3).astype(np.uint8)
    plt.scatter(bb[0], bb[1], color='red', s = 100)
    plt.imshow(img)
    
val = get_data(path+'valid', load_size)
show_bb(98)
# check whats going wrong here


log.info('Read in data')
if not input_exists:

    batches = get_batches(path+'train', batch_size=batch_size)
    val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)
    (val_classes, trn_classes, val_labels, trn_labels, 
        val_filenames, filenames, test_filenames) = get_classes(path)
    
    # Fetch our large images 
    log.info('Fetch images')
    trn = get_data(path+'train', load_size)
    val = get_data(path+'valid', load_size)
    test = get_data(path+'test', load_size)
    
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
    #lrg_model[i].compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    lrg_model[i].compile(Adam(lr=0.0001), loss=['mse'], metrics=['mse'])
    lrg_model[i].fit(conv_trn_feat, trn_bbox, batch_size=batch_size, nb_epoch=2,
                 validation_data=(conv_val_feat, val_bbox))             
    lrg_model[i].optimizer.lr=1e-7
    lrg_model[i].fit(conv_trn_feat, trn_bbox, batch_size=batch_size, nb_epoch=6,
                 validation_data=(conv_val_feat, val_bbox))

    # Make our prediction on the lrg_model layer
    log.info('Output Prediction')
    predsls.append(lrg_model[i].predict(conv_test_feat, batch_size=batch_size)) # or try 32 batch_size
    pvalsls.append(lrg_model[i].predict(conv_val_feat, batch_size=batch_size))
    val_score = "%.3f" % metrics.log_loss(val_labels, sum(pvalsls)/len(pvalsls))
    acc_score = "%.3f" % accuracyfunc(val_labels, do_clip(sum(pvalsls)/len(pvalsls), clip))
    log.info('Bagged Validation Logloss ' + str(val_score))
    log.info('Bagged Validation Accuracy ' + str(acc_score))

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

