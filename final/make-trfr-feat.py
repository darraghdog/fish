# coding: utf-8

# In[1]:

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
# get_ipython().magic(u'matplotlib inline')

def accuracyfunc(y_act, y_pred):
    return metrics.accuracy_score(np.argmax(y_act, axis=1), np.argmax(y_pred, axis=1))
    
def refresh_directory_structure(name, sub_dirs):
    gdir = os.path.join(path, name)
    if os.path.exists(gdir):
        shutil.rmtree(gdir)
    os.makedirs(gdir)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(gdir, sub_dir))


# In[2]:

# Set Parameters and check files
refresh_directories =    False # True
input_exists =           True  # False 
full =                   True 
log.info('Set Paramters')
path =       "../data/fish/"
batch_size=  32
clip =       0.99
bags =       40           # 20
load_size =  (440, 780)  # (360, 640)


# In[3]:

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
        


# In[4]:

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


# In[5]:

# Read in the boxes
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
bb_json = {}
for c in anno_classes:
    j = json.load(open(os.path.join(path, 'box/{}_labels.json'.format(c)), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]

# make it easy to find the nof dots, by putting themin the middle
#empty_bbox = {'height': 0., 'width': 0., 'x': 1280/2., 'y': 720/2}
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox


# In[6]:

# Finally, we convert the dictionary into an array, and convert the coordinates to our resized 224x224 images.
bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (load_size[1] / size[0])#(224. / size[0])
    conv_y = (load_size[0] / size[1])#(224. / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb


# In[7]:

trn_sizes = [PIL.Image.open(path+'train/'+f).size for f in filenames]
val_sizes = [PIL.Image.open(path+'valid/'+f).size for f in val_filenames]
tst_sizes = [PIL.Image.open(path+'test/'+f).size for f in test_filenames]


# In[8]:

trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, trn_sizes)], 
                   ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s) 
                   for f,s in zip(raw_val_filenames, val_sizes)]).astype(np.float32)


# In[9]:

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(i):
    bb = val_bbox[i]
    plot(val[i])
    plt.gca().add_patch(create_rect(bb))

#del val
gc.collect()


# In[10]:

log.info('Read in data')
if not input_exists:

    batches = get_batches(path+'train', batch_size=batch_size)
    val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)
    (val_classes, trn_classes, val_labels, trn_labels, 
        val_filenames, filenames, test_filenames) = get_classes(path)
    
    # Fetch our large images 
    # Precompute the output of the convolutional part of VGG
    log.info('Fetch images')
    log.info('Get VGG output')
    log.info('Write VGG output')
    
    val = get_data(path+'valid', load_size)
    conv_val_feat = vgg640.predict(val, batch_size=16, verbose=1)
    save_array(path+'results/1_conv_val_big_feat.dat', conv_val_feat)
    del val, conv_val_feat
    gc.collect()
    
    trn = get_data(path+'train', load_size)
    conv_trn_feat = vgg640.predict(trn, batch_size=16, verbose=1)    
    del trn
    gc.collect()
    save_array(path+'results/1_conv_trn_big_feat.dat', conv_trn_feat) 
    del conv_trn_feat
    gc.collect()
    
    test = get_data(path+'test', load_size)
    conv_test_feat = vgg640.predict(test, batch_size=16, verbose=1)
    save_array(path+'results/1_conv_test_big_feat.dat', conv_test_feat)     
    del test, conv_test_feat
    gc.collect()

    # For memory purposes delete out the original train and validation
    log.info('Clear up memory')
    #del trn, val, test
    gc.collect()
    gc.collect()

# Start script 2

    
def refresh_directory_structure(name, sub_dirs):
    gdir = os.path.join(path, name)
    if os.path.exists(gdir):
        shutil.rmtree(gdir)
    os.makedirs(gdir)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(gdir, sub_dir))


# Set Parameters and check files
refresh_directories = True  # False
input_exists =        False # True
full =                True
log.info('Set Paramters')
path =                "../data/fish/"
batch_size=           32
clip =                0.99
bags =                50  
load_size =           (380, 680) # (360, 640) 

relabels = pd.read_csv("../data/fish/relabel/relabels.csv", sep = " ", header = None, names = ["fname", "dir_from", "dir_to"])
subdir2 = relabels[relabels.fname == 'img_00739.jpg'.split('.')[0]].values.tolist()[0][2]
subdir2 == 'revise'

# Create the test and valid directory
if refresh_directories:
    log.info('Create directory structure and validation files')
    sub_dirs = os.listdir(os.path.join(path, 'train-all'))
    if '.DS_Store' in sub_dirs: sub_dirs.remove('.DS_Store')
    refresh_directory_structure('relabel/train', sub_dirs)
    refresh_directory_structure('relabel/valid', sub_dirs)
    for c,row in enumerate(csv.DictReader(open('../image_validation_set.csv'))):
        value = 'relabel/valid' if row['Validation'] == '1' else 'relabel/train'
        subdir1 = row['SubDirectory']
        subdir2 = row['SubDirectory']
        if row['file_name'].split('.')[0] in relabels.fname.values.tolist():
            subdir2 = relabels[relabels.fname == row['file_name'].split('.')[0]].values.tolist()[0][2]
            print(relabels[relabels.fname == row['file_name'].split('.')[0]].values.tolist())
        name_from = os.path.join(path, 'train-all', subdir1, row['file_name'])
        if subdir2 == 'revise':
            name_to   = os.path.join(path, 'relabel/revise', row['file_name'])
            #print(os.path.join(path, value, subdir2, row['file_name']))
        else:
            name_to   = os.path.join(path, value, subdir2, row['file_name'])
            #print(os.path.join(path, value, subdir2, row['file_name']))
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
    val_filenames, filenames, test_filenames) = get_classes(path+"relabel/")

# Read in filenames
log.info('Read filenames')
raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]

folder_anno_in = 'darknet/FISH/annos'
# Read in the boxes
anno_classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
tmpdict = {}
for c in anno_classes:
    j = json.load(open(os.path.join('..', folder_anno_in, '{}.json'.format(c)), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            tmpdict[l['filename'].split('/')[-1]] = sorted(
               l['annotations'], key=lambda x: x['height']*x['width'])[-1]

bb_json = {k: tmpdict[k] for k in raw_filenames + raw_val_filenames if k in tmpdict}

# make it easy to find the nof dots, by putting themin the middle
#empty_bbox = {'height': 0., 'width': 0., 'x': 1280/2., 'y': 720/2}
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox

bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (load_size[1] / size[0])
    conv_y = (load_size[0] / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb

trn_sizes = [PIL.Image.open(path+'relabel/train/'+f).size for f in filenames]
val_sizes = [PIL.Image.open(path+'relabel/valid/'+f).size for f in val_filenames]
tst_sizes = [PIL.Image.open(path+'relabel/test/'+f).size for f in test_filenames]


# In[11]:

sizes = [PIL.Image.open(path+'relabel/train/'+f).size for f in filenames]
raw_val_sizes = [PIL.Image.open(path+'relabel/valid/'+f).size for f in val_filenames]
trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)], 
                   ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s) 
                   for f,s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)


# In[12]:

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(i):
    bb = val_bbox[i]
    plot(val[i])
    plt.gca().add_patch(create_rect(bb))

#val = get_data(path+'relabel/valid', load_size)
#show_bb(500)
#del val
#gc.collect()


# In[13]:

log.info('Read in data')
if not input_exists:    
    log.info('Validation - Fetch images; Get VGG output; Write VGG output')    
    val = get_data(path+'relabel/valid', load_size)
    conv_val_feat = vgg640.predict(val, batch_size=16, verbose=1)
    save_array(path+'results/2_conv_val_relabel_feat.dat', conv_val_feat)
    del val, conv_val_feat
    gc.collect()
    
    log.info('Train - Fetch images; Get VGG output; Write VGG output')
    trn = get_data(path+'relabel/train', load_size)
    conv_trn_feat = vgg640.predict(trn, batch_size=16, verbose=1)    
    del trn
    gc.collect()
    save_array(path+'results/2_conv_trn_relabel_feat.dat', conv_trn_feat) 
    del conv_trn_feat
    gc.collect()
    
    log.info('Test - Fetch images; Get VGG output; Write VGG output')
    test = get_data(path+'relabel/test', load_size)
    conv_test_feat = vgg640.predict(test, batch_size=16, verbose=1)
    save_array(path+'results/2_conv_test_relabel_feat.dat', conv_test_feat)     
    del test, conv_test_feat
    gc.collect()



