import argparse

import math
import numpy as np
import os
import random
import json
import re
import datetime
import time

from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D, MaxPooling2D, Merge
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K

from keras.callbacks import EarlyStopping

# =============================================================================

parser = argparse.ArgumentParser()

# positional arguments (required)
parser.add_argument('path_root', type=str, help="path to 'datasets' directory")
parser.add_argument('dataset', type=str, help='name of the dataset. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('p', type=str, help='p parameter of node2vec. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('q', type=str, help='q parameter of node2vec. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('definition', type=int, help='definition. E.g., 14 for 14:1. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('n_channels', type=int, help='number of channels. Must not exceed half the depth of the tensors in tensors/*dataset*/node2vec_hist/ folder')

# optional arguments
parser.add_argument('--n_folds', type=int, default=10, choices=[2,3,4,5,6,7,8,9,10], help='number of folds for cross-validation')
parser.add_argument('--n_repeats', type=int, default=3, choices=[1,2,3,4,5], help='number of times each fold should be repeated')
parser.add_argument('--batch_size', type=int, default=32, choices=[32,64,128], help='batch size')
parser.add_argument('--nb_epochs', type=int, default=50, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=5, help='patience for early stopping strategy')
parser.add_argument('--drop_rate', type=float, default=0.3, help='dropout rate')

args = parser.parse_args()

# convert command line arguments
path_root = args.path_root
dataset = args.dataset
p = args.p
q = args.q
definition = args.definition
n_channels = args.n_channels

n_folds = args.n_folds
n_repeats = args.n_repeats
batch_size = args.batch_size
nb_epochs = args.nb_epochs
my_patience = args.patience
drop_rate = args.drop_rate

dim_ordering = 'th' # channels first
my_optimizer = 'adam'
#my_optimizer = 'adamax'
# command line examples: python main.py /home/antoine/Desktop/graph_2D_CNN/datasets/ imdb_action_romance 1 1 14 5
#                        python main.py /home/antoine/Desktop/graph_2D_CNN/datasets/ imdb_action_romance 1 1 14 5 --n_folds 10 --n_repeats 1 --nb_epochs 20 --patience 3

# =============================================================================

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

# =============================================================================

def main():
    
    fold_name = ''
    histo_name = ''
    my_date_time = ''
    
    name_save = path_root + '/results/' + dataset + '_augmentation_' + my_date_time
      
    print ('========== loading labels ==========')

    with open(path_root + 'classes/' + dataset + '/' + dataset + '_classes.txt', 'r') as f:
        ys = f.read().splitlines()
        ys = [int(elt) for elt in ys]

    num_classes = len(list(set(ys)))

    print ('========== conducting', n_folds ,'fold cross validation =========='); 
    print ('repeating each fold:', n_repeats, 'times')
    
    fold_data = np.load(path_root+'/results/backup/'+fold_name)
    folds = fold_data['folds']
    folds_labels = fold_data['labels']
    shuffled_idxs = fold_data['shuffled_idxs']
    input_shape = fold_data['input_shape']
    fold_data.close()
    print ('fold sizes:', [len(fold) for fold in folds])
    
    history_data = np.load(path_root+'/results/backup/'+histo_name) 
    outputs = history_data['outputs']
    histories = history_data['histories']
    history_data.close()
    
    fold_exsit = len(histories)

    for i in range(fold_exsit+1,n_folds):
        
        t = time.time()
        
        x_train = np.concatenate([fold for j,fold in enumerate(folds) if j!=i],axis=0)
        x_test = [fold for j,fold in enumerate(folds) if j==i]
        
        y_train = np.concatenate([y for j,y in enumerate(folds_labels) if j!=i],axis=0)
        y_test = [y for j,y in enumerate(folds_labels) if j==i]
        
        for repeating in range(n_repeats):
            
            print ('clearing Keras session')
            K.clear_session()
            
            my_input = Input(shape=input_shape, dtype='float32')
            
            conv_1 = Convolution2D(64,
                                   3,
                                   3,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_1 = MaxPooling2D(pool_size=(2,2),
                                         dim_ordering=dim_ordering
                                         )(conv_1)

            pooled_conv_1_dropped = Dropout(drop_rate)(pooled_conv_1)
            
            conv_11 = Convolution2D(96,
                                    3,
                                    3,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_1_dropped)
            
            pooled_conv_11 = MaxPooling2D(pool_size=(2,2),
                                          dim_ordering=dim_ordering
                                          )(conv_11)
                                          
            pooled_conv_11_dropped = Dropout(drop_rate)(pooled_conv_11)
            pooled_conv_11_dropped_flat = Flatten()(pooled_conv_11_dropped)

            conv_2 = Convolution2D(64,
                                   4,
                                   4, 
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_2 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_2)
            pooled_conv_2_dropped = Dropout(drop_rate)(pooled_conv_2)
            
            conv_22 = Convolution2D(96,
                                    4,
                                    4, 
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering,
                                    )(pooled_conv_2_dropped)
            
            pooled_conv_22 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_22)
            pooled_conv_22_dropped = Dropout(drop_rate)(pooled_conv_22)
            pooled_conv_22_dropped_flat = Flatten()(pooled_conv_22_dropped)

            conv_3 = Convolution2D(64,
                                   5,
                                   5,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_3 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_3)
            pooled_conv_3_dropped = Dropout(drop_rate)(pooled_conv_3)
            
            conv_33 = Convolution2D(96,
                                    5,
                                    5,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_3_dropped)
            
            pooled_conv_33 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_33)
            pooled_conv_33_dropped = Dropout(drop_rate)(pooled_conv_33)
            pooled_conv_33_dropped_flat = Flatten()(pooled_conv_33_dropped)                        
            
            conv_4 = Convolution2D(64,
                                   6,
                                   6,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_4 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_4)
            pooled_conv_4_dropped = Dropout(drop_rate)(pooled_conv_4)
            
            conv_44 = Convolution2D(96,
                                    6,
                                    6,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_4_dropped)
            
            pooled_conv_44 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering) (conv_44)
            pooled_conv_44_dropped = Dropout(drop_rate) (pooled_conv_44)
            pooled_conv_44_dropped_flat = Flatten()(pooled_conv_44_dropped)

            merge = Merge(mode='concat')([pooled_conv_11_dropped_flat,
                                          pooled_conv_22_dropped_flat,
                                          pooled_conv_33_dropped_flat,
                                          pooled_conv_44_dropped_flat])
            
            merge_dropped = Dropout(drop_rate)(merge)
            
            dense = Dense(128,
                          activation='relu'
                          )(merge_dropped)
            
            dense_dropped = Dropout(drop_rate)(dense)
            
            prob = Dense(output_dim=num_classes,
                         activation='softmax'
                         )(dense_dropped)
            
            # instantiate model
            model = Model(my_input,prob)
                            
            # configure model for training
            model.compile(loss='categorical_crossentropy',
                          optimizer=my_optimizer,
                          metrics=['accuracy'])
            
            print ('model compiled')
            
            early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as acc on validation set increases
                                           patience=my_patience,
                                           mode='max') 
            
            history = model.fit(x_train,
                                y_train,
                                batch_size=batch_size,
                                nb_epoch=nb_epochs,
                                validation_data=(x_test, y_test),
                                callbacks=[early_stopping])

            
            # save [min loss,max acc] on test set
            max_acc = max(model.history.history['val_acc'])
            max_idx = model.history.history['val_acc'].index(max_acc)
            output = [model.history.history['val_loss'][max_idx],max_acc]
            outputs.append(output)
            
            # also save full history for sanity checking
            histories.append(model.history.history)
            
            #find out the incorrect predictions
            y_prob = model.predict(x_test)
            y_predictions = y_prob.argmax(axis=-1)
            y_classes = np.nonzero(y_test[0])[1]
            incorrects = np.nonzero(y_predictions != y_classes)[0]
            
            # find out the original index of x test
            shuffle_dict = [x+int(i*(len(shuffled_idxs)/n_folds)) for x in range(int(len(shuffled_idxs)/n_folds))]
            original_idx = shuffled_idxs[shuffle_dict]
          
            print ('incorrect ratio %.3f'%(incorrects.size/y_predictions.size))
#            incorrect_label = y_predictions[incorrects]
#            correct_label = y_classes[incorrects]
            incorrect_predictions = np.vstack((y_classes,y_predictions))
            
            # save model
            save_name = dataset + '_fold_%d_repeating_%d'%(i,repeating)+my_date_time
            model.save('models/%s.h5'%save_name)
            np.savez_compressed('incorrect/%s'%(save_name+'_incorrect_predictions'), ic = incorrects,ip=incorrect_predictions, oi = original_idx)
          
            # save temp results to disk
            np.savez_compressed(path_root + '/results/backup/' + dataset+ my_date_time+ '_histories',outputs = outputs,histories=histories)    
        
        print ('**** fold', i+1 ,'done in ' + str(math.ceil(time.time() - t)) + ' second(s) ****')

    # save results to disk
    with open(name_save + '_results.json', 'w') as my_file:
        json.dump({'outputs':outputs,'histories':histories}, my_file, sort_keys=False, indent=4)

    print( '========== results saved to disk ==========')

if __name__ == "__main__":
    main()
