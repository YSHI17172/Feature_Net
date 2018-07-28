from keras.models import load_model
import numpy as np
from keras import backend as K
import os

model_path = '/Users/User/OneDrive - University of South Carolina/New/Feature_Net/models/'

features = ['slot','step','boss','pocket','pyramid',
            'protrusion','through_hole','blind_hole','cone',]

data_path = '/Users/User/Downloads/graph_2D_CNN/datasets/tensors/nine/node2vec_hist/'
file_name = 'nine_10:1_p=2_q=0.5_'

nine = []
sample_size = 100
for i in range(9):
    sample = np.random.randint(1000,size=sample_size)+(1000*i)
    ftr = []
    for smp in sample:
        ftr.append(np.load(data_path+file_name+str(smp)+'.npy'))
    nine.append(np.array(ftr))
nine = np.array(nine)
positive = nine[[2,4,5,8]]
negative = nine[[0,1,3,6,7]]

#predict_path = '/Users/User/Downloads/graph_2D_CNN/datasets/tensors/predict/node2vec_hist/'
#k45 = []
#k45_files= os.listdir(predict_path)
#try:
#    k45_files.remove('.DS_Store') # read all file names in the folder
#except:
#    pass
#for f45 in k45_files:
#    x = np.load(predict_path+f45)
#    k45.append(x[:,:41,:41])
#
#k45 = np.array(np.split(np.array(k45),9))
#sample_size = 500
#positive = k45[[2,4,5,8]]
#negative = k45[[0,1,3,6,7]]

neg_predictions = [[] for i in range(5)]

negative_model = load_model(model_path+'negative2.h5')
for negind,neg_set in enumerate(negative):
    prob_result = negative_model.predict(neg_set)
    label_result = prob_result.argmax(axis=-1)
    neg_predictions[negind].append(label_result)
del(negative_model)
K.clear_session()

slotNstep = load_model(model_path+'slotNstep.h5')
for negind,neg_set in enumerate(negative):
    prob_result = slotNstep.predict(neg_set)
    label_result = prob_result.argmax(axis=-1)
    neg_predictions[negind].append(label_result)
del(slotNstep)
K.clear_session()

pocketNhole = load_model(model_path+'pocketNhole.h5')
for negind,neg_set in enumerate(negative):
    prob_result = pocketNhole.predict(neg_set)
    label_result = prob_result.argmax(axis=-1)
    neg_predictions[negind].append(label_result)
del(pocketNhole)
K.clear_session()    

hole = load_model(model_path+ 'hole.h5')
for negind,neg_set in enumerate(negative):
    prob_result = hole.predict(neg_set)
    label_result = prob_result.argmax(axis=-1)
    neg_predictions[negind].append(label_result)
del(hole)
K.clear_session()    

neg_predictions = [np.array(f) for f in neg_predictions]

predict_label = [[] for i in range(5)]
for fpred_ind,f_predict in enumerate(neg_predictions):
    for i in range(f_predict.shape[1]):
        samp = f_predict[:,i]
        if samp[0] == 0:
            if samp[1] == 0:
                label = 0 #slot
            elif samp[1] ==1:
                label = 1 #step
        elif samp[0] == 1:
            if samp[2] == 0:
                label = 3 #pocket
            elif samp[2] == 1:
                if samp[3] == 0:
                    label = 6 #blind_hole
                elif samp[3] == 1:
                    label = 7 #through_hole
        predict_label[fpred_ind].append(label)
predict_label = [np.array(f) for f in predict_label]
for i,n in enumerate([0,1,3,6,7]):
    incorrects = np.nonzero(predict_label[i] != n)[0]
    print ('feature %d loss rate: %.2f'%(n,incorrects.size/sample_size))

pos_predictions = [[] for i in range(4)]

pos_model = load_model(model_path+'positive2.h5')
for posind,pos_set in enumerate(positive):
    prob_result = pos_model.predict(pos_set)
    label_result = prob_result.argmax(axis=-1)
    pos_predictions[posind].append(label_result)
del(pos_model)
K.clear_session()

bossPconeNpy = load_model(model_path+'bossPconeNpy.h5')
for posind,pos_set in enumerate(positive):
    prob_result = bossPconeNpy.predict(pos_set)
    label_result = prob_result.argmax(axis=-1)
    pos_predictions[posind].append(label_result)
del(bossPconeNpy)
K.clear_session()

bossNcone = load_model(model_path+'bossNcone.h5')
for posind,pos_set in enumerate(positive):
    prob_result = bossNcone.predict(pos_set)
    label_result = prob_result.argmax(axis=-1)
    pos_predictions[posind].append(label_result)
del(bossNcone)
K.clear_session()

pos_predictions = [np.array(f) for f in pos_predictions]

predict_label_pos = [[] for i in range(4)]
for fpred_ind,f_predict in enumerate(pos_predictions):
    for i in range(f_predict.shape[1]):
        samp = f_predict[:,i]
        if samp[0] == 0:
            if samp[1] == 1:
                label = 4 # pyramid
            elif samp[1] == 0:
                if samp[2]== 0:
                    label = 2 #boss
                elif samp[2] == 1:
                    label = 8 #cone                
        elif samp[1] == 1:
            label = 5 #protrusion
        predict_label_pos[fpred_ind].append(label)
predict_label_pos = [np.array(f) for f in predict_label_pos]
for i,n in enumerate([2,4,5,8]):
    incorrects = np.nonzero(predict_label_pos[i] != n)[0]
    print ('feature %d loss rate: %.2f'%(n,incorrects.size/sample_size))
