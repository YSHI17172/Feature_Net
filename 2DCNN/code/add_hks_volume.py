# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil


def normlize(v,maxx,minn):
    my_max = np.amax(v)
    my_min = np.amin(v)
    norm = minn + (v-my_min)*(maxx-minn)/(my_max-my_min)
    return norm

features = ['slot','step','boss','pocket','pyramid',
        'protrusion','through_hole','blind_hole','cone',]


dataset = 'DEMO04_95'

cluster_path = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/%s/cluster/'%dataset
emb_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/%s/'%dataset
new_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/%s_att/'%dataset
#check = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/check/'

head1 = '%s_node2vec_raw_p=1_q=1_'%dataset
#head2 = 'new_negative_5k_node2vec_raw_p=4_q=0.25_'

#list file name version
files = os.listdir(emb_path)
for f in files:
    try:
        volume_hks = np.load(cluster_path+f[len(head1):])
        
        n2v1 = np.load(emb_path+f)
        vmax1 = np.amax(n2v1)
        vmin1 = np.amin(n2v1)
        volume_rate1 = normlize(volume_hks[0],vmax1,vmin1)
        hks1 = normlize(volume_hks[1],vmax1,vmin1)
        new1 = np.column_stack((volume_rate1,hks1,n2v1))
        np.save(new_path+f,new1,allow_pickle=False)
    except Exception as e:
        print(f,e)
        print(volume_rate1.shape,hks1.shape,n2v1.shape)
    

# for i in range(5000):
#     try:
#         volume_hks = np.load(cluster_path+str(i)+'.npy')
#         
#         n2v1 = np.load(emb_path+head1+str(i)+'.npy')
#         vmax1 = np.amax(n2v1)
#         vmin1 = np.amin(n2v1)
#         volume_rate1 = normlize(volume_hks[0],vmax1,vmin1)
#         hks1 = normlize(volume_hks[1],vmax1,vmin1)
#         new1 = np.column_stack((volume_rate1,hks1,n2v1))
#         np.save(new_path+head1+str(i),new1,allow_pickle=False)
#     except:
#         pass
#     
    #n2v2 = np.load(emb_path+head2+str(i)+'.npy')
    # vmax2 = np.amax(n2v2)
    # vmin2 = np.amin(n2v2)
    # volume_rate2 = normlize(volume_hks[0],vmax2,vmin2)
    # hks2 = normlize(volume_hks[1],vmax2,vmin2)
    # new2 = np.column_stack((volume_rate2,hks2,n2v2))
    # np.save(new_path+head2+str(i),new2,allow_pickle=False)
    # 
    
    
#     elif n2v.shape[0] +2 == n2v.shape[1]:
#         os.rename(emb_path+head+str(i)+'.npy',new_path+head+str(i)+'.npy')
#     else:
#         os.rename(emb_path+head+str(i)+'.npy',check+head+str(i)+'.npy')
#         