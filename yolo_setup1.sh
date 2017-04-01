# Yolo Set up - Part 1
# If you are stuck on anything refer to the instructions here : https://pjreddie.com/darknet/yolo/

# From main directory, run the below
git clone https://github.com/pjreddie/darknet
cd darknet
make

# Yolo has examples of VOC data training. We will download this and modify for our training
curl -O http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
curl -O http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
curl -O http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# The below is for generating the labels for VOC images; again we use this as a baseline.
curl -O http://pjreddie.com/media/files/voc_label.py
python voc_label.py

# Now we extract the weights file we shall use for training on the fishes.
curl -O http://pjreddie.com/media/files/darknet19_448.conv.23

# Now we do some specifics for our training
mkdir darknet/FISH/
mkdir darknet/FISH/JPEGImages
mkdir darknet/FISH/labels
mkdir darknet/FISH/annos
python voc_label_FISH1.py

