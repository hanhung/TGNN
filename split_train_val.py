import os, sys, glob

DIR = './preprocessed_scannet/'

# Copy files to train folder
train_dir = './train/'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

with open('ScanRefer/ScanRefer_filtered_train.txt') as f:
    scenes = f.readlines()
scenes = [x.strip() + '.pth' for x in scenes]

for scene in scenes:
    command = 'cp ' + DIR + scene + ' ' + train_dir
    os.system(command)

# Copy files to val folder
val_dir = './val/'
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

with open('ScanRefer/ScanRefer_filtered_val.txt') as f:
    scenes = f.readlines()
scenes = [x.strip() + '.pth' for x in scenes]

for scene in scenes:
    command = 'cp ' + DIR + scene + ' ' + val_dir
    os.system(command)
