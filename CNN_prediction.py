from keras.models import load_model
import numpy as np
from keras import backend as K
import os

def get_hist_node2vec(emb,d,my_min,my_max,img_dim):
    my_bins = np.linspace(my_min,my_max,img_dim+1) #  to have middle bin centered on zero
    Hs = []
    for i in range(0,d,2):
        H, xedges, yedges = np.histogram2d(x=emb[:,i],y=emb[:,i+1],bins=my_bins, normed=False)
        Hs.append(H)
    Hs = np.array(Hs)    
    return  Hs

model_path = '/Volumes/MAC256/models/new_negative_5k_c20/c5/'


features = ['slot/','step/','pocket/','through/','blind/']

dataset = 'intersecting_c20'
path_to_node2vec = '/Users/User/Downloads/graph_2D_CNN/datasets/raw_node2vec/%s/'%dataset
file_name = '%s_node2vec_raw_p=1_q=1_'%dataset
file_names = [file_name + '%d.npy'%i for i in range(150000)]

max_n_channels = 5

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

img_dim = 55
#my_min = -2.9
#my_max = 2.6
samples = []
for name in file_names:
    emb = np.load(path_to_node2vec + name)
    tensor = get_hist_node2vec(emb,n_dim,my_min,my_max,img_dim)
    samples.append(tensor)
samples = np.array(samples)

#samples = []
#sample_size = 1000
#for i in range(len(features)):
#    sample = np.random.randint(1000,size=sample_size)+(1000*i)
#    ftr = []
#    for smp in sample:
#        emb = np.load(path_to_node2vec+file_name+str(smp)+'.npy')
#        tensor = get_hist_node2vec(emb,n_dim,my_min,my_max,img_dim)
#        ftr.append(tensor)
#    samples.append(np.array(ftr))
#samples = np.array(samples)

predictions = []

model_files = os.listdir(model_path)
model_files = [f for f in model_files if '.h5' in f]

for model_ind,model_fname in enumerate(model_files):
    negative_model = load_model(model_path+model_fname)
    prob_result = negative_model.predict(samples)
    predictions.append(prob_result)   
    del(negative_model)
    K.clear_session() 
    print ('model %d finished'%model_ind)

np.savez_compressed('/Volumes/MAC1T/intersecting_predicted_by_negative_c1_original_normalized',predictions=predictions)    
    
#models = [load_model(model_path+model_fname) for model_fname in model_files]
#for negative_model in models:
#    model_prediction = [[] for i in range(5)]
#    for ind,sample_set in enumerate(samples):
#        prob_result = negative_model.predict(sample_set)
#        label_result = prob_result.argmax(axis=-1)
#        model_prediction[ind].append(label_result)
#    predictions.append(model_prediction)

original_label = np.array([[i]*30000 for i in range(5)]).flatten()

for n in range(30):
    predict_label = predictions[n].argmax(axis=-1)
    accuracy = np.count_nonzero(predict_label == original_label) / 1500
    print ('Overall accuracy %.4f%% by model %d'%(accuracy,n))
    
    for i in range(5):
        feature_label = predict_label[i*30000:(i+1)*30000]
        original_feature_label = original_label[i*30000:(i+1)*30000]
        class_accuracy = np.count_nonzero(feature_label == original_feature_label) / 300       
        print ('Feature %d accuracy : %.4f'%(i,class_accuracy))
   
    print ('--------------------------------------------------')

average_prob = predictions[0]
for n in range(1,30):
    average_prob+= predictions[n]
average_prob/= 30.
predict_label = average_prob.argmax(axis=-1)
accuracy = np.count_nonzero(predict_label == original_label) / 1500
print ('Average accuracy %.4f%%'%(accuracy))  

for i in range(5):
    feature_label = predict_label[i*30000:(i+1)*30000]
    original_feature_label = original_label[i*30000:(i+1)*30000]
    class_accuracy = np.count_nonzero(feature_label == original_feature_label) / 300       
    print ('Feature %d accuracy : %.4f'%(i,class_accuracy))  
    print ('--------------------------------------------------')


        
        
        
