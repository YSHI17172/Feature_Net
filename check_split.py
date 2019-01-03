# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import feature_extraction as fe

def run(f_name,model_path,hks_path,path):
    data = np.load(model_path+f_name+'.npz')
    tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
    mesh_model = data['my_model'][()] ; step_model = data['step_model'][()]
    data.close()
    
    hks_data = np.load(hks_path+f_name+'.npz')  
    hks_persistence = hks_data['persistence']
     #adjpts = hks_data['adjpts'][()]
    hks_data.close()
    
    if hks_persistence.shape[0] != coord_array.shape[0]:
        print ('#######%s,model和hks不匹配,分别是%d %d#########'\
        %(f_name,hks_persistence.shape[0],coord_array.shape[0])) 
    
    if len(mesh_model.features) > 1:
        print ('******模型%s共有%d个feature*******'%(f_name,len(mesh_model.features)))  
        return 0
    
    feature_model = fe.feature_model(step_model,mesh_model,hks_persistence)
      
    if len(feature_model.features) != 2:
        print ('分解后%s共有%d个feature'%(f_name,len(feature_model.features)))
        for feature in feature_model.features:
            print (feature.type,feature.faces)
        return 1
    # else:
    #     # tttt = [feature.type for feature in feature_model.features]
    #     # if not ('BLIND' in tttt and 'THROUGH' in tttt):
    #     #     for feature in feature_model.features:
    #     #         print (feature.type,feature.faces)
    #     #         print ('-----------------------')
    #     if not len(feature_model.features[0].faces)>1 and \
    #         not len(feature_model.features[0].faces)>1:
    #             print ('%s分解后面不足'%(f_name))
    #             for feature in feature_model.features:
    #                 print (feature.type,feature.faces)

    
    # # get feature info
    # feature_model = fe.solid_model(step_model,mesh_model,hks_persistence)
    # 
    # print ('%s共有%d个feature'%(f_name,len(feature_model.features)))
    # print (feature_model.features)
    # for feature in feature_model.features:
    #     print (feature.faces)


    #print ('Part %s finished'%f_name)
    #end
if __name__ == '__main__':
    import os
    import sys
    
    features1 = ['through_hole_through_hole','through_hole_blind_hole','blind_hole_blind_hole',\
    'blind_hole_pocket','through_hole_pocket','pocket_pocket','blind_hole_slot',\
    'through_hole_slot','slot_pocket','slot_slot','blind_hole_step','pocket_step',\
    'through_hole_step','slot_step','step_step']
    features2 = ['through_hole1_through_hole2','through_hole_blind_hole','blind_hole1_blind_hole2',\
    'blind_hole_pocket','through_hole_pocket','pocket1_pocket2','blind_hole_slot',\
        'through_hole_slot','slot_pocket','slot1_slot2','blind_hole_step','pocket_step',\
        'through_hole_step','slot_step','step1_step2']
    features3 = [['through','through'],['through','blind'],['blind','blind'],\
    ['pocket','blind'],['pocket','through'],['pocket','pocket'],['slot','blind'],\
        ['slot','through'],['slot','pocket'],['slot','slot'],['step','blind'],['step','pocket'],\
        ['step','through'],['step','slot'],['step','step']]
    
    i = 8
    feature1 = features1[i]
    feature2 = features2[i]
    
    save_path = '/Volumes/ExFAT256/adjm_cluster/'
    hks_path = '/Volumes/MAC256/clusters/%s/'%feature1
    model_path ='/Volumes/ExFAT256/intersecting_models/%s/'%feature1
    #model_path = '/Volumes/ExFAT256/new/blind_hole_pocket/'
    adjm_path = '/Volumes/ExFAT256/adjm_cluster/%s/'%feature1
    
    files = [feature2 + '_' + str(i) for i in range(0,5000)]
        
    a = os.listdir(adjm_path+features3[i][0])
    b = os.listdir(adjm_path+features3[i][1])
    
    feature_a = [int(f[len(feature2)+1:-5]) for f in a if '_' in f]
    feature_b = [int(f[len(feature2)+1:-5]) for f in b if '_' in f]
    
    error = []
    for n in range(5000):
        if not n in feature_a and not n in feature_b:
            in1 = n in feature_a
            in2 = n in feature_b
            #print ('%s %d 有问题，%s %s'%(feature2,n,in1,in2))
            error.append(n)
    print(error)

    files = [feature2 + '_' + str(i) for i in error]
    
    check = []
    for fid,fname in enumerate(files):
        #print(fname)
        aaa = run(fname,model_path,hks_path,save_path)
        if aaa == 1:
            check.append(error[fid])
    
    # n_jobs = cpu_count()-1
    # to_parallelize_partial = partial(run,model_path=model_path,hks_path=hks_path,path=save_path)
    # pool = Pool(processes=n_jobs)
    # pool.map(to_parallelize_partial,files)
    # pool.close()
    # pool.join()
    #         