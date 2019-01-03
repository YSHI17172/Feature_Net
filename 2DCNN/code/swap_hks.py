import numpy as np


path = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/new_isolated_10k_c20/cluster/'
save_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/new_isolated_10k_c20_cluster/'


total = 100000

# for i in range(10):
#     for j in range(5000,total):
#         old_n = i*total + j
#         new_n = 40000*i + j
# 
#         a = np.load(path+'%d.npy'%old_n)
#         b = a.swapaxes(0,1)
#         head = 'intersecting_c20_modified+10k_cluster_node2vec_raw_p=1_q=1_%d'%(new_n)
#         np.save(save_path+head,b)


for j in range(total):
    a = np.load(path+'%d.npy'%j)
    b = a.swapaxes(0,1)
    head = 'new_isolated_10k_c20_cluster_node2vec_raw_p=1_q=1_%d'%(j)
    np.save(save_path+head,b)