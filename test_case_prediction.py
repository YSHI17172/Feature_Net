from keras.models import load_model
import numpy as np
from keras import backend as K
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def get_hist_node2vec(emb,d,my_min,my_max,img_dim):
    my_bins = np.linspace(my_min,my_max,img_dim+1) #  to have middle bin centered on zero
    Hs = []
    for i in range(0,d,2):
        H, xedges, yedges = np.histogram2d(x=emb[:,i],y=emb[:,i+1],bins=my_bins, normed=False)
        Hs.append(H)
    Hs = np.array(Hs)    
    return  Hs

model_path_c1 = '/Users/User/OneDrive - University of South Carolina/New/Feature_Net/2DCNN/models/c1/'
model_path_c5 = '/Users/User/OneDrive - University of South Carolina/New/Feature_Net/2DCNN/models/c5/'


features = ['slot/','step/','pocket/','through/','blind/']

dataset = 'sipe_socket_1'
path_to_node2vec = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/%s/'%dataset
file_names = os.listdir(path_to_node2vec)
file_names.sort(key=natural_keys)
head = '%s_node2vec_raw_p=1_q=1_'%dataset
try:
    file_names.remove('.DS_Store')
except:
    pass

max_n_channels = 1

# load tensors, get max min
tensors = []
n_dim = 2*max_n_channels
for idx, name in enumerate(file_names):
    tensor = np.load(path_to_node2vec + name)
    tensors.append(tensor[:,:n_dim])
    if len(file_names) > 10 and\
    idx % round(len(file_names)/10) == 0:
        print (idx)
print ('tensors loaded')

full = np.concatenate(tensors)
my_max = np.amax(full)
my_min = np.amin(full)
print ('range:', my_max, my_min)
del(full)
del(tensors)

img_dim = 56
#my_min = -2.6465;my_max = 2.9524 #new_negative_5k_c20
#my_min = -2.607;my_max = 3.081 #intersecting_c20_modified
#my_min =  -2.145662092976376;my_max = 2.6473773370516454 #DEMO04_1
my_min =  -1.7594328796796723;my_max = 2.4553299647200157 #sipe_socket_1

samples = []
for name in file_names:
    emb = np.load(path_to_node2vec + name)
    tensor = get_hist_node2vec(emb,n_dim,my_min,my_max,img_dim)
    samples.append(tensor)
samples = np.array(samples)


predictions = []

model_files = os.listdir(model_path_c1)
model_files = [f for f in model_files if '.h5' in f]

for model_ind,model_fname in enumerate(model_files):
    negative_model = load_model(model_path_c1+model_fname)
    prob_result = negative_model.predict(samples)
    predictions.append(prob_result)   
    del(negative_model)
    K.clear_session() 
    print ('model %d finished'%model_ind)

np.savez_compressed('/Users/User/OneDrive - University of South Carolina/New/Feature_Net/2DCNN/predictions/%s'%dataset,predictions=predictions)    

#predictions = np.load('/Users/User/OneDrive - University of South Carolina/New/Feature_Net/2DCNN/predictions/%s.npz'%dataset)['predictions']

original_label = []
for name in file_names:
    number = int(name[len(head):len(head)+1])-1
    original_label.append(number)
original_label = np.array(original_label).flatten()

#for n in range(30):
#    predict_label = predictions[n].argmax(axis=-1)
#    accuracy = np.count_nonzero(predict_label == original_label) /len(file_names)
#    print ('Overall accuracy %.4f%% by model %d'%(accuracy,n))
#    for f in range(len(file_names)):       
#        print (f,original_label[f],predict_label[f])
#        #print (predictions[n][f])
#    print ('--------------------------------------------------')

average_prob = predictions[0]
for n in range(1,30):
    average_prob+= predictions[n]
average_prob/= 30.
predict_label = average_prob.argmax(axis=-1)
accuracy = np.count_nonzero(predict_label == original_label) /len(file_names)
print ('Average accuracy %.4f%%'%(accuracy))  

for i in range(5):    
    for f in range(len(file_names)): 
        if original_label[f] ==i: #and 'g' in file_names[f][-9:-4]:
            shape = np.load(path_to_node2vec+file_names[f]).shape[0]
        
            print (f,file_names[f][-9:-4],shape,original_label[f],predict_label[f])
            #print (average_prob[f])
    print ('--------------------------------------------------')


        
        
        
