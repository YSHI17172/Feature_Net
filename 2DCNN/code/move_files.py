# -*- coding: utf-8 -*-
import os

d = '/Users/User/Downloads/graph_2D_CNN/datasets/tensors/hole2/node2vec_hist/'

features = ['slot','step','boss','pocket','pyramid',
            'protrusion','through_hole','blind_hole','cone',]


files= os.listdir(d)

try:
    files.remove('.DS_Store') # read all file names in the folder
except:
    pass

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

# for f in files:
#     number = f[20:-4]
#     if 5999<int(number)<7000 :
#         os.rename(d+f, 'tensors/trough_hole/'+f)

for f in files[:]:
#    new = f[:12] + ':' +f[13:]
#    
     try:
         number = int(f[20:-4])
     except ValueError:
         print (f)
         
     if number > 5000:
         nn = number -6000
         new = f[:20] + str(nn)+'.npy'
         os.rename(d+f, d+new)

#    new = 'negative2' + f[9:]
#    os.rename(d+f, d+new)

    