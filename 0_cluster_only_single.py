# -*- coding: utf-8 -*-
import generate_hks
import persistence
import connection_matrix_point
import numpy as np
import os
import sys

def run(f_name,path,save_path):
    data = np.load(path+f_name)
    tri_array = data['tri_array']
    coord_array = data['coord_array']
    data.close()

    # Calculate HKS
    steps = 0.001
    iters = 1000
    HKS = generate_hks.generate_hks(coord_array, tri_array, steps, iters)

    # Calculate persistence
    v = persistence.persistence(HKS)

    # get the connection matrix of points, and save in file 'connection_matrix_points.npy'
    adjpts = connection_matrix_point.connection_matrix(coord_array,tri_array)

    save_name = f_name[:-4]
    np.savez_compressed(save_path+save_name, persistence = v,adjpts=adjpts)
    #end

features = ['through_hole_through_hole','through_hole_blind_hole','blind_hole_blind_hole',\
        'blind_hole_pocket','through_hole_pocket','pocket_pocket','blind_hole_slot',\
        'through_hole_slot','slot_pocket','slot_slot','blind_hole_step','pocket_step',\
        'through_hole_step','slot_step','step_step']
features2 = ['through_hole1_through_hole2','through_hole_blind_hole','blind_hole1_blind_hole2',\
        'blind_hole_pocket','through_hole_pocket','pocket1_pocket2','blind_hole_slot',\
        'through_hole_slot','slot_pocket','slot1_slot2','blind_hole_step','pocket_step',\
        'through_hole_step','slot_step','step1_step2']

feature_index = 4
feature = features[feature_index]
feature2 = features2[feature_index]

#path = '/data/userdata/yshi/intersecting/%s/'%feature
#save_path = '/data/userdata/yshi/clusters/%s/'%feature

path = '/Volumes/ExFAT256/new/%s/'%feature
save_path = '/Volumes/ExFAT256/clusters/%s/'%feature

to_do_number = [302, 1710, 2169, 4694]

for fn in to_do_number:

    fname = '%s_%d.npz'%(feature2,fn)
    run(fname,path,save_path)
