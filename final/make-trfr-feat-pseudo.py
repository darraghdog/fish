
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
from distutils.dir_util import copy_tree
import glob

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
refresh_directories = True
input_exists = False
full = True
log.info('Set Paramters')
path = "../data/fish/"
batch_size=32
clip = 0.99
bags = 50
load_size = (400, 700)#(440, 780)#(360, 640)
yolo_cutoff = 0.7
sub_cutoff = 0.8

# Create the test and valid directory
if refresh_directories:
    log.info('Create directory structure and validation files')
    sub_dirs = os.listdir(os.path.join(path, 'train-all'))
    if '.DS_Store' in sub_dirs: sub_dirs.remove('.DS_Store')
    refresh_directory_structure('pseudo/train', sub_dirs)
    refresh_directory_structure('pseudo/valid', sub_dirs)
    refresh_directory_structure('pseudo/test', ['test'])
    for c,row in enumerate(csv.DictReader(open('../image_validation_set.csv'))):
        value = 'pseudo/valid' if row['Validation'] == '1' else 'pseudo/train'
        name_from = os.path.join(path, 'train-all', row['SubDirectory'], row['file_name'])
        name_to   = os.path.join(path, value, row['SubDirectory'], row['file_name'])
        shutil.copyfile(name_from, name_to)   
    copy_tree(os.path.join(path, 'test'), os.path.join(path, 'pseudo/test'))

# Use our best some to idenify high confidence test images
best_sub = pd.read_csv("../sub/final-round1-input-pseudo.csv")
hiconf_test = best_sub[best_sub.drop(['image'], axis=1).apply(np.max, axis=1)>sub_cutoff].reset_index(drop=True)
hiconf_test['class'] = hiconf_test.drop(['image'], axis=1).idxmax(axis = 1)

# Now we read in our high confidence boundary boxes. 
# Load up YOLO bounding boxes for each class
all_files = glob.glob(os.path.join('../yolo_coords', "*.txt"))
allFiles = [f for f in all_files if 'FISH544.txt' in f]
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=None, sep = " ", names = ['image', 'proba', 'x0', 'y0', 'x1', 'y1'])
    df['class'] = file_.split('_')[-1].split('.')[0]
    list_.append(df)
yolo_frame = pd.concat(list_)
yolo_frame = yolo_frame[yolo_frame['proba']>yolo_cutoff].sort(['image'])
# Just keep the highest confidence one
yolo_frame = yolo_frame.loc[yolo_frame.groupby(['image'])['proba'].idxmax()].reset_index(drop=True)
yolo_frame['image'] = yolo_frame['image'] + '.jpg'


# In[7]:

# Get the innder join of each as we need the labels and the boundary box
yolo_frame = yolo_frame[yolo_frame['image'].isin(hiconf_test['image'].tolist())].reset_index(drop=True)
hiconf_test = hiconf_test[(hiconf_test['image'].isin(yolo_frame['image'].tolist())) | (hiconf_test['class']=='NoF')].reset_index(drop=True)
hiconf_test.shape, yolo_frame.shape


# In[8]:

# Now we do pseudo labelling by copying test images to pseudo/train data set
for i in range(len(hiconf_test)):
    row = hiconf_test.iloc[i].values.tolist()
    img = row[0]
    img_class = row[9]
    name_from = os.path.join(path, 'test', 'test', img)
    name_to   = os.path.join(path, 'pseudo/train', img_class, img)
    shutil.copyfile(name_from, name_to) 
    #print (img, img_class)


# In[9]:

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
    val_filenames, filenames, test_filenames) = get_classes(path+ 'pseudo/')

# Read in filenames
log.info('Read filenames')
raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]


# In[10]:

# Read in the boxes
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
bb_json = {}
for c in anno_classes:
    j = json.load(open(os.path.join(path, 'box/{}_labels.json'.format(c)), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]

# In[12]:
for i in range(len(yolo_frame)):
    row = yolo_frame.iloc[i].values.tolist()
    image, x, y, width, height = row[0], row[2], row[3], row[4]-row[2], row[5]-row[3]
    bb_json[image] = {}
    bb_json[image]['class'] = 'rect'
    bb_json[image]['x'] = x
    bb_json[image]['y'] = y
    bb_json[image]['height'] = height
    bb_json[image]['width'] = width


# In[13]:

# make it easy to find the nof dots, by putting themin the middle
#empty_bbox = {'height': 0., 'width': 0., 'x': 1280/2., 'y': 720/2}
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox


# In[14]:

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


# In[15]:
trn_sizes = [PIL.Image.open(path+'pseudo/train/'+f).size for f in filenames]
val_sizes = [PIL.Image.open(path+'pseudo/valid/'+f).size for f in val_filenames]
tst_sizes = [PIL.Image.open(path+'pseudo/test/'+f).size for f in test_filenames]


# In[16]:
trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, trn_sizes)], 
                   ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s) 
                   for f,s in zip(raw_val_filenames, val_sizes)]).astype(np.float32)


# In[17]:

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(i):
    bb = trn_bbox[i]
    plot(trn[i])
    plt.gca().add_patch(create_rect(bb))

# In[18]:

log.info('Read in data')
if not input_exists:
    batches = get_batches(path+'pseudo/train', batch_size=batch_size)
    val_batches = get_batches(path+'pseudo/valid', batch_size=batch_size*2, shuffle=False)
    (val_classes, trn_classes, val_labels, trn_labels, 
        val_filenames, filenames, test_filenames) = get_classes(path+ 'pseudo/')
    
    # Fetch our large images 
    # Precompute the output of the convolutional part of VGG
    log.info('Fetch images')
    log.info('Get VGG output')
    log.info('Write VGG output')
    
    val = get_data(path+'pseudo/valid', load_size)
    conv_val_feat = vgg640.predict(val, batch_size=16, verbose=1)
    save_array(path+'results/6_conv_val_pseudo_feat.dat', conv_val_feat)
    del val, conv_val_feat
    gc.collect()
    
    trn = get_data(path+'pseudo/train', load_size)
    conv_trn_feat = vgg640.predict(trn, batch_size=16, verbose=1)    
    del trn
    gc.collect()
    save_array(path+'results/6_conv_trn_pseudo_feat.dat', conv_trn_feat) 
    del conv_trn_feat
    gc.collect()
    
    test = get_data(path+'pseudo/test', load_size)
    conv_test_feat = vgg640.predict(test, batch_size=16, verbose=1)
    save_array(path+'results/6_conv_test_pseudo_feat.dat', conv_test_feat)     
    del test, conv_test_feat
    gc.collect()

    # For memory purposes delete out the original train and validation
    log.info('Clear up memory')
    #del trn, val, test
    gc.collect()
    gc.collect()
