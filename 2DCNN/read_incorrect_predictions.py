import numpy as np
import os
import sys
import matplotlib.pyplot as plt
#from mayavi import mlab

os.chdir(sys.path[0])

path = 'incorrect/new_isolated_c20_cluster_d30/'

features =['Slot','Step','Pocket','Through_hole','Blind_hole',
            'Boss','Pyramid','Protrusion','Cone','Dome']   
            
#features = ['Boss','Pyramid','Protrusion','Cone','Dome'] #positive
#features = ['Slot','Step','Pocket','Through_hole','Blind_hole',] #negative


files= os.listdir(path)

for f in files[:]:
    if 'incorrect' not in f:
        files.remove(f)       

fn = len(features) ; fold = 10;sn =5000;repeat = 3

total = np.zeros((fn,fn))
wrong_matrix = [[[] for i in range(fn)] for i in range(fn)]
together = []
total2 = []
for fi,f in enumerate(files):

    data = np.load(path+f)
    predictions = data['ip']
    incorrects = data['ic']
    original_idx = data['oi']
    labels = predictions[:,incorrects]
    wrong_idx = original_idx[incorrects]
    data.close()
    
    #print(predictions.shape,incorrects.shape,original_idx.shape)
    stats = np.zeros((fn,fn))    
    for w in range(labels.shape[1]):
        i = labels[0,w] # original label
        j = labels[1,w] # prediction
        stats[i,j] +=1 
        wrong_matrix[i][j].append(wrong_idx[w])
        
    num_each = []
    for i in range(fn):
        num = np.count_nonzero(predictions[0,:]==i)#test sample number
        num_each.append(num)
        stats[i,:] /= num
    together.append(stats)
    #print (len(wrong_matrix[4][3]))
    total += stats
    total2.append((original_idx.size-incorrects.size)/original_idx.size)

total /= repeat*fold
print ('Totoal Loss',np.mean(total2)*100,np.std(total2)*100)

for ftind,ft in enumerate(features):
    values = total[ftind,:]
    vi = np.argsort(-values)
    print ('**********************')
    print (ft+' statistics:')
 
    total_repeats = [1-np.sum(rept[ftind,:]) for rept in together]
    print ('total loss',np.mean(total_repeats)*100,np.std(total_repeats)*100)
    
    for vidx,vii in enumerate(vi):
        if values[vii] > 0:
            print (features[vii] + ': %.5f'%(values[vii]))
    #print ('wrong index', wrong_matrix[ftind])
    print ('---------------------')        

#np.save('/Volumes/MAC1T/wrong_matrix_intersecting_c20_channel_5_reapt_1',wrong_matrix)
# for i,f in enumerate(wrong_matrix):
# 
#         w = []
#         for ff in f:
#             w += ff
#         w = np.unique(w)
#         w -= 30000*i
#         reshape = np.zeros(30000)
#         reshape[w.astype(np.int)] = 1
#         fig = plt.figure()
#         hist = np.histogram(w,bins=[0,5000,10000,15000,20000,25000,30000,35000])
#         plt.hist(w)
#         #plt.bar(np.arange(30000),reshape)
#         print(w.size,hist)
    
#plt.show()
        



