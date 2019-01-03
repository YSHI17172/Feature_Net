# -*- coding: utf-8 -*-
import argparse
import os
import re
import numpy as np
import time as t

def main():
    t_start = t.time()
    
    path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/new_isolated_10k_c20_cluster/'
    
    all_file_names  = os.listdir(path) 
    print ('===== total number of files in folder: =====', len(all_file_names))
    try:
        all_file_names.remove('.DS_Store')
    except:
        pass
    # load tensors
    tensors = []
    for idx, name in enumerate(all_file_names):
        tensor = np.load(path + name)
        #tensors.append(tensor.swapaxes(0,1))
        tensors.append(tensor[:,:2])
        if len(all_file_names) > 10 and\
        idx % round(len(all_file_names)/10) == 0:
            print (idx)
    
    print ('tensors loaded')
    
    full = np.concatenate(tensors)
    my_max = np.amax(full)
    my_min = np.amin(full)   
    print ('range:', my_max, my_min)

    for d in range(20):
        definition = (d+1)*10
        img_dim = int(np.arange(my_min, my_max+0.05,(my_max+0.05-my_min)/float(definition*(my_max+0.05-my_min))).shape[0]-1)

        print('definition: ',definition,'dimension',img_dim)

if __name__ == "__main__":
    main()