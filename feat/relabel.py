# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:28:07 2017

@author: darragh
"""

import os
import shutil
os.chdir("/home/darragh/Dropbox/fish/feat")

TRAIN_PATH = "../data/fish/relabel"
TRAIN_PATH_IN  = "../data/fish/relabel/train"
TRAIN_PATH_OUT = "../data/fish/revise"
RELABELS_PATH = "relabels.csv"

try:
    os.mkdir("{}/{}".format(TRAIN_PATH, "revise"))
except:
    print("already created")


with open(RELABELS_PATH) as f:
    for line in f:
        cols = line.split()
        src = "{}/{}/{}.jpg".format(TRAIN_PATH_IN, cols[1], cols[0])
        dst = "{}/{}_from_{}_to_{}.jpg".format(TRAIN_PATH_OUT, cols[0], cols[1], cols[2])

        try:
            # os.rename(src, dst)
            shutil.copy(src, dst)

        except FileNotFoundError:
            print("{} not found".format(src))   