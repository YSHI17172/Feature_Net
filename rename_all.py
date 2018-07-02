# -*- coding: utf-8 -*-
import os

d = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/adjm/'

features = ['slot','step','boss','pocket','pyramid',
            'protrusion','through_hole','blind_hole','cone',]  


files= os.listdir(d)

# e = os.listdir('/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/test')
# 
# m = []
# for r in e:
#     m.append(r[-9:-4])
#     
# missing = []
# 
# for f in files:
#     if f[:-4] not in m:
#         print (f)
#         missing.append(f)
#     # else: #放进t 文件夹
#     #     os.rename(d+f, d+'t/'+f)
#         
# print (len(missing))

# v = 'adjm/'
# adjm = os.listdir(v)
# new = []
# for a in adjm:
#     new.append(a[:-4])
# 
# for f in files:
#     if  f in new:
#         os.rename(d+f,d+'t/'+f)
c =0
for f in files:
    if f[0:4] == 'cone':
        if f[-7:-4] == 'npz':
            number = f[5:-8]
        else:
            number = f[5:-4]
        new = int(number)+8000
        print (new)
        c+=1
        os.rename(d+f,d+str(new)+'.txt')