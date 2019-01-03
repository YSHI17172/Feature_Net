import shutil
import os
import numpy as np
import random

adjm_path = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/intersecting_c20/'

remove = np.load('/Volumes/MAC1T/wrong_matrix_intersecting_c20_channel_5_reapt_1.npy')

tensor_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/intersecting_c20/'

new_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/intersecting_c20_modified/'

total = 30000

f = 4
w = []
for ff in remove[f]:
    w += ff
w = np.unique(w)

for i in range(total):
    i = i+total*f
    name = 'intersecting_c20_node2vec_raw_p=1_q=1_%d.npy'%(i)
    if i not in w:
        shutil.copyfile(tensor_path+name,new_path+name)
    else:
        new = 0
        shape = 0
        random.seed()
        while new in w or shape>40 or shape < 25:
            new = random.randint(total*f,total*f+10000)
            adjm = np.loadtxt(adjm_path+'%d.txt'%new)
            shape = adjm.shape[0]
        print(i,new,shape)
        copy = 'intersecting_c20_node2vec_raw_p=1_q=1_%d.npy'%new
        shutil.copyfile(tensor_path+copy,new_path+name)