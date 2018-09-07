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
    hks_persistence = hks_data['persistence'];adjpts = hks_data['adjpts']
    hks_data.close()
    
    if hks_persistence.shape[0] != coord_array.shape[0]:
        print (f_name,hks_persistence.shape[0],coord_array.shape[0])
    
    # get feature info
    #feature_model = fe.feature_model(step_model,mesh_model,hks_persistence)
    
    # print ('%s共有%d个feature'%len(feature_model.features))
    # print (feature_model.features)
    # for feature in feature_model.features:
    #     print (feature.faces)

    # # Find clusters
    # simil = 0.85
    # adjm = [np.array([]),np.array([])]
    # while adjm[0].shape[0]<10 and adjm[1].shape[0]<10 and simil <= 0.995:
    #     clusters = cluster.cluster(coord_array, tri_array, hks_persistence, adjpts,simil)
    #     cluster_adjmap = cluster.get_attributed_cluster_adj_matrix(simil,clusters,tri_array)
    #     adjm,fcluster = save_adj.get_feauture_cluster_and_adjm(feature_model,mesh_model,cluster_adjmap,clusters)
    #     simil += 0.005
    #   
    # ab = ['a','b']
    # for fid,feature in enumerate(feature_model.features):
    #   
    #     if feature.type == 'BLIND':
    #         save_path = path + 'blind/'
    #     elif feature.type == 'THROUGH':
    #         save_path = path + 'through/'
    #             
    #     save_name = f_name + ab[fid]
    #     np.savetxt(save_path+save_name,adjm[fid],fmt='%d')
    #     np.save(save_path+'cluster/'+save_name,fcluster[fid])
    
    #print ('Part %s finished'%f_name)
    #end
if __name__ == '__main__':
    import os
    import sys
    os.chdir(sys.path[0])
    
    feature2 = 'through_hole_blind_hole'
    feature1 = 'through_hole_blind_hole'
    
    
    save_path = 'adjm_cluster/'
    hks_path = '/Users/ting/Downloads/clusters/%s/'%feature1
    model_path = 'input_mesh/intersecting/%s/'%feature1
    model_path ='/Users/ting/Downloads/through_hole_blind_hole/'
    
    files = [feature2 + '_' + str(i) for i in range(100,1000)]
    
    for fname in files:
        run(fname,model_path,hks_path,save_path)
    
    # n_jobs = cpu_count()-1
    # to_parallelize_partial = partial(run,mesh_path=mesh_path,path=save_path)
    # pool = Pool(processes=n_jobs)
    # pool.map(to_parallelize_partial,files)
    # pool.close()
    # pool.join()
    #         


