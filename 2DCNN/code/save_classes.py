import numpy as np
import os

#clear = open('test_classes.txt','w')
#clear.close()

path = '/Users/User/Downloads/graph_2D_CNN/datasets/classes/'

os.chdir(path)

cls = np.array([[i]*10000 for i in range(10)]).flatten()

#cls = [0]*1000+[1]*1000

cls = np.array(cls).flatten()

name = 'new_isolated_10k_c20_cluster'

new_path = path + name
try:
    os.mkdir(new_path)
except FileExistsError:
    pass
np.savetxt('%s/%s_classes.txt'%(new_path,name),cls,fmt = '%d')