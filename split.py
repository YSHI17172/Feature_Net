# -*- coding: utf-8 -*-
import cluster
import numpy as np
import save_adj
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
    adjpts = hks_data['adjpts'][()]
    hks_data.close()
    
    if hks_persistence.shape[0] != coord_array.shape[0]:
        print ('%s points number not matching %d,%d'%(f_name,hks_persistence.shape[0],coord_array.shape[0]))
        return None

    if len(mesh_model.features) > 1:
        print ('mesh模型%s共有%d个feature'%(f_name,len(mesh_model.features)))  
        return 0
    
    feature_model = fe.feature_model(step_model,mesh_model,hks_persistence)
    
    #if len(feature_model.features) != 2:
    print ('分解后%s共有%d个feature'%(f_name,len(feature_model.features)))
    for feature in feature_model.features:
        print (feature.type,feature.faces)
     #   return None
    # elif feature_model.features[0].type != 'THROUGH'\
    #   or feature_model.features[1].type != 'THROUGH':
    #     print ('%s有feature类型错误!'%f_name)
    #     for feature in feature_model.features:
    #         print (feature.type,feature.faces)
    # 
        # Find clusters
    simil = 0.85
    adjm = [np.array([]),np.array([])]
    while (adjm[0].shape[0]<10 or adjm[1].shape[0]<10) and simil <= 0.995:
        clusters = cluster.cluster(coord_array, tri_array, hks_persistence,adjpts,simil)
        cluster_adjmap = cluster.get_attributed_cluster_adj_matrix(simil,clusters,tri_array)
        adjm,fcluster = save_adj.get_feauture_cluster_and_adjm(feature_model,mesh_model,cluster_adjmap,clusters)
        simil += 0.005
      
    ab = ['a','b']
    for fid,feature in enumerate(feature_model.features):
      
        if feature.type == 'BLIND':
            save_path = path + 'blind/'
        elif feature.type == 'THROUGH':
            save_path = path + 'through/'
        elif feature.type == 'POCKET':
            save_path = path + 'pocket/'
        elif feature.type == 'SLOT':
            save_path = path + 'slot/'
        elif feature.type == 'STEP':
            save_path = path + 'step/'
            
        save_name = f_name + ab[fid]
        np.savetxt(save_path+save_name+'.txt',adjm[fid],fmt='%d')
        np.save(save_path+'cluster/'+save_name,fcluster[fid])
    
    print(simil,adjm[0].shape,adjm[1].shape)
    print ('Part %s finished'%f_name)
    #end
if __name__ == '__main__':
    import os
    import sys
    os.chdir(sys.path[0])
    
    features1 = ['through_hole_through_hole','through_hole_blind_hole','blind_hole_blind_hole',\
    'blind_hole_pocket','through_hole_pocket','pocket_pocket','blind_hole_slot',\
    'through_hole_slot','slot_pocket','slot_slot','blind_hole_step','pocket_step',\
    'through_hole_step','slot_step','step_step']
    features2 = ['through_hole1_through_hole2','through_hole_blind_hole','blind_hole1_blind_hole2',\
    'blind_hole_pocket','through_hole_pocket','pocket1_pocket2','blind_hole_slot',\
        'through_hole_slot','slot_pocket','slot1_slot2','blind_hole_step','pocket_step',\
        'through_hole_step','slot_step','step1_step2']
    features3 = ['through/cluster/','through/cluster/','blind/cluster/',\
    'pocket/cluster/','pocket/cluster/','pocket/cluster/','slot/cluster/',\
        'slot/cluster/','slot/cluster/','slot/cluster/','step/cluster/','step/cluster/',\
        'step/cluster/','step/cluster/','step/cluster/']
    
    i = 8
    feature1 = features1[i]
    feature2 = features2[i]
    
    save_path = '/Volumes/ExFAT256/adjm_cluster/%s/'%feature1
    hks_path = '/Volumes/MAC256/clusters/%s/'%feature1
    model_path ='/Volumes/ExFAT256/intersecting_models/%s/'%feature1
    
    print('start checking %s'%feature1)
    check = [104]
    for fn in check:
        fname = '%s_%d'%(feature2,fn)
        run(fname,model_path,hks_path,save_path)
    
    # n_jobs = cpu_count()-1
    # to_parallelize_partial = partial(run,mesh_path=mesh_path,path=save_path)
    # pool = Pool(processes=n_jobs)
    # pool.map(to_parallelize_partial,files)
    # pool.close()
    # pool.join()
    #         