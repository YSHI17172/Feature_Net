# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
import re

path = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/history/four5/'

files= os.listdir(path)

for f in files[:]:
    if '.' not in f:
        files.remove(f)       
try:
    files.remove('.DS_Store')
except:
    pass

save_path = '/Users/User/Downloads/mesh_temp/'

mesh_path = '/Users/User/OneDrive - University of South Carolina/New/Feature_Net/input_mesh/'


c = 0
for f in files[:]:
    adjm = np.loadtxt(path+f)
    shape = adjm.shape
    if shape[1]< 10:
        # print('')
        # print ('这个也有问题！！！！！！！！！！！ %s'%f)
        # print('')
        fname = f[:-4]
        fnumber = int(re.search(r'_(\d*)',f).group(1))
        ftype = re.search(r'(\w*)_\d*',f).group(1)
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
        shutil.copyfile(mesh_path+sub_folder+ftype+'/'+fname+'.npz',save_path+fname+'.npz')
        c+=1
        
    if shape[0] != shape[1]:
        if shape[0] == 2*shape[1]:
            new = adjm[:shape[1],:shape[1]]
            np.savetxt(path+f,new,fmt='%d')  
            print('搞定一个! %s'%f)
            c +=1
        else:
            print('')
            print ('这个有问题！！！！！！！！！！！ %s'%f)
            print('')

print (c)
    