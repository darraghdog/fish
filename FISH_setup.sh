# Before starting git clone darknet 
cd darknet
mkdir FISH
cd FISH
mkdir labels
git clone https://github.com/autoliuweijie/Kaggle
mkdir annos
cp Kaggle/NCFM/datasets/* annos/
rm -rf Kaggle
