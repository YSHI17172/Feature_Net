# -*- coding: utf-8 -*-
from read_STEP import read_STEP,closed_shell
import geometry_builder as gb
import datetime
import numpy as np
import re

def mesh(model_path,mesh_length=1,model_only=False,
        sub_type=None):

    start_time = datetime.datetime.now()
    
    min_length = mesh_length*1.5 # lower length won't be seperated
    
    fname = re.findall(r'/(.+_\d+).step',model_path)[0]
    data = read_STEP(model_path) 
    step_model = closed_shell(data)
    pts_array,facets,hole_facets = step_model.get_facets(mesh_length=mesh_length)
       
    #generate model2
    my_model = gb.solid_model(pts_array,facets,hole_facets,min_length)
    
    if model_only == False:
        #generate mesh
        coord_array,tri_array = my_model.generate_mesh(mesh_length=mesh_length)
    else:
        coord_array = tri_array = []
        
    end_time = datetime.datetime.now()
    elapsed = (end_time-start_time).seconds
    #print ("%s Mesh Created, has %d points, taken time %d seconds.\n"
    #%(fname,coord_array.shape[0],elapsed))
    
    return step_model,my_model,coord_array,tri_array,

def run(i,feature,path,save_path):
    fname = '%s_%d.step'%(feature,i)    
    save_name = fname[:-5]
    try:
        step_model,my_model,coord_array,tri_array = mesh(path+fname,mesh_length = 1)
        np.savez_compressed(save_path+save_name, step_model=step_model, my_model=my_model,
        coord_array=coord_array,tri_array=tri_array.astype(np.int))
    except:
        print('check %s'%fname)

if __name__ == "__main__": 
    import os
    import sys
    #import re
    from multiprocessing import Pool, cpu_count
    from functools import partial
    os.chdir(sys.path[0])
    
    features = ['through_hole_through_hole','through_hole_blind_hole','blind_hole_blind_hole',\
    'blind_hole_pocket','through_hole_pocket','pocket_pocket','blind_hole_slot',\
    'through_hole_slot','slot_pocket','slot_slot','blind_hole_step','pocket_step',\
    'through_hole_step','slot_step','step_step']
    features2 = ['through_hole1_through_hole2','through_hole_blind_hole','blind_hole1_blind_hole2',\
    'blind_hole_pocket','through_hole_pocket','pocket1_pocket2','blind_hole_slot',\
        'through_hole_slot','slot_pocket','slot1_slot2','blind_hole_step','pocket_step',\
        'through_hole_step','slot_step','step1_step2']
        
    #for feature_index in range(1,15):
    feature_index = 8
    feature = features[feature_index]
    feature2 = features2[feature_index]
    path = 'STEP/%s/'%feature#test sample path
    #save_path = 'input_mesh/intersecting/%s/'%feature
    #save_path = '/Volumes/ExFAT256/intersecting/%s/'%feature
    save_path = '/Volumes/ExFAT256/intersecting_models/%s/'%feature
    #save_path = '/Volumes/ExFAT256/new/'
    
    #exsiting_files= os.listdir(save_path)
    #name_len = len(feature2)+1
    #exsiting_numbers = [int(name[name_len:-4]) for name in exsiting_files if '_' in name]    
    #to_do_number = [n for n in range(5000) if n not in exsiting_numbers]
    #print(to_do_number)

    to_do_number = [620]

    n_jobs = 8
    to_parallelize_partial = partial(run,feature=feature2,path=path,save_path=save_path)
    pool = Pool(processes=n_jobs)
    pool.map(to_parallelize_partial,to_do_number)
    pool.close()
    #pool.join()
        
    # for number in to_do_number[:]:
    #     fname = '%s_%d.step'%(feature,number)   
    #     #fname = 'blind_hole1_blind_hole2_%d.step'%(number)          
    #     save_name = fname[:-5]
    #     #print(fname)
    #     #step_model,my_model,coord_array,tri_array = mesh(path+fname)
    #     try:
    #         step_model,my_model,coord_array,tri_array = mesh(path+fname)
    #         np.savez_compressed(save_path+save_name, step_model=step_model, my_model=my_model,
    #         coord_array=coord_array,tri_array=tri_array.astype(np.int))
    #     except:
    #         print('check %s'%fname)
    #         continue

    # from mayavi import mlab
    # mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
    # #mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2],)
    # mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
    #     tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
    # mlab.show()
