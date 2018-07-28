import generate_hks
import persistence
import cluster
import connection_matrix_point
import numpy as np
import os
import sys
import save_adj
import datetime

start_time = datetime.datetime.now()

os.chdir(sys.path[0]) #change dir to main's path

features = ['slot','step','boss','pocket','pyramid',
        'protrusion','through_hole','blind_hole','cone',]    

#feature_index = 1
#feature_type = features[feature_index]

for feature_type in features[:-1]:

    files= os.listdir('input_mesh/'+feature_type);
    try:
        files.remove('.DS_Store') # read all file names in the folder
    except:
        pass

    for counter,f_name in enumerate(files):
        part_time = datetime.datetime.now()
        data = np.load('input_mesh/'+feature_type+'/'+f_name)
        model = data['model'][()]
        tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
        data.close()
        
        print ("Part %s hase %d points, and %d triangles in mesh.\n"%(f_name,np.shape(coord_array)[0], np.shape(tri_array)[0]))
        
        # Calculate HKS
        steps = 0.001
        iters = 1000
        HKS = generate_hks.generate_hks(coord_array, tri_array, steps, iters)
    
        # Calculate persistence
        v = persistence.persistence(HKS)
        
        # get the connection matrix of points, and save in file 'connection_matrix_points.npy'
        connection_matrix_point.connection_matrix(coord_array,tri_array)
        
        # Find clusters
        simil = 0.85
        adjm = np.array([])
        while adjm.shape[0]<16:
            clusters = cluster.cluster(coord_array, tri_array, v, simil)    
            cluster_adjmap = cluster.get_attributed_cluster_adj_matrix(simil,clusters,tri_array)
            adjm = save_adj.get_feauture_cluster_adjm(model,cluster_adjmap,clusters)
            simil += 0.01
        end_time = datetime.datetime.now()
        elapsedtime = (end_time - part_time).seconds
        total_time = (end_time - start_time).seconds
        print ('Adjacency Matrix has %d clusters in there.'%adjm.shape[0])
        print ('Part %d finished, taken time %s seconds, total elapsedtime is %.1f minutes.\n'
                %(counter,elapsedtime,total_time/60))
        save_adj.save_txt(f_name,adjm)    
        