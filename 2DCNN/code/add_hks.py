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

adjm_path = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/four5/'
cluster_path = '/Users/User/Downloads/adjm_cluster/cluster/'
mesh_path = '/Users/User/OneDrive - University of South Carolina/New/Feature_Net/input_mesh/'
save_path = '/Users/User/Downloads/mesh_error/'

vectors = []
missing = []
c=0
for i in range(45000):
    adjm = np.loadtxt(adjm_path+str(i)+ '.txt')
    v = np.diagonal(adjm)
    if np.sum(v) == 0:
        try:
            v = np.load(cluster_path+str(i)+'.npy')[1,:]
        except:
            c+=1
            missing.append(i)
    vectors.append(v)

for m in missing:
    fnumber = m%5000
    type_number = m//5000
    ftype = features[type_number]
    if fnumber < 2000:
        sub_folder = 'mesh_1k/'
    elif 3000>fnumber>1999:
        sub_folder = 'mesh_2k/'
    elif 4000>fnumber>2999:
        sub_folder = 'mesh_3k/'
    elif 5000>fnumber>3999:
        sub_folder = 'mesh_4k/'
    else:
        print('读取错误,%s'%f)
    fname = ftype + '_' +str(fnumber)+'.npz'
    shutil.copyfile(mesh_path+sub_folder+ftype+'/'+fname,save_path+fname)

emb_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/four5/'
new_path = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/four5att/'

efiles =  os.listdir(emb_path)
try:
    efiles.remove('.DS_Store') # read all file names in the folder
except:
    pass

for name in efiles[:]:
    try:
        number = int(name[29:-4])
    except ValueError:
          print (name)
    n2v = np.load(emb_path+name)
    vmax = np.amax(n2v)
    vmin = np.amin(n2v)
    norm = normlize(vectors[number],vmax,vmin)
    if n2v.shape[0] != norm.shape[0]:
         shutil.copyfile(adjm_path+str(number)+'.txt',
         '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/error/'+str(number)+'.txt')
    else:
        new = np.column_stack((norm,n2v))
        np.save(new_path+name[:-4],new,allow_pickle=False)
