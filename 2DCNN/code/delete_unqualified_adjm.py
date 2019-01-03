import numpy as np
import os

emb_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/intersecting/'
cluster_path = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/intersecting_cluster/'
new_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/intersecting_att/'
    
head = 'intersecting_node2vec_raw_p=2_q=0.5_'

check = []
for i in range(150000):
    n2v = np.load(emb_path+head+str(i)+'.npy')
    #volume_hks = np.load(cluster_path+str(i)+'.npy')
    if n2v.shape[1] < 10:
        print(i,n2v.shape)
        check.append(i)
        os.remove(emb_path+head+str(i)+'.npy')
        os.remove(new_path+head+str(i)+'.npy')