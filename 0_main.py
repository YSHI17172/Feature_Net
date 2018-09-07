import read_dat
import plot_mesh
import generate_hks
import plot_hks
import persistence
import plot_persistence
import cluster_old
import plot_clusters
import connection_matrix_point
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

os.chdir(sys.path[0]) #change dir to main's path  

fname = 4894
from read_STEP import read_STEP,closed_shell
import geometry_builder as gb
feature2 = 'through_hole_blind_hole'
feature1 = 'through_hole_blind_hole'
step_path = 'STEP/%s/%s_%d.step'%(feature1,feature2,fname)
step_data = read_STEP(step_path) 
step_model = closed_shell(step_data)
coo_array,facets,hole_facets = step_model.get_facets()
my_model = gb.solid_model(coo_array,facets,hole_facets,min_length=1.5)

coord_array,tri_array = my_model.generate_mesh(mesh_length=1)
tri_array=tri_array.astype(np.int)
save_path = 'input_mesh/intersecting/%s_%d'%(feature2,fname)
np.savez_compressed(save_path, step_model=step_model, my_model=my_model,
coord_array=coord_array,tri_array=tri_array)

# part_name = "input_mesh/Boat2.dat"
# coord_array, tri_array = read_dat.read_dat(part_name)

# data = np.load('input_mesh/intersecting_research/blind_hole&pocket/blind_hole&pocket_4.npz')
# tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
# data.close()

print ("This part hase %d points, and %d triangles in mesh.\n"%(np.shape(coord_array)[0], np.shape(tri_array)[0]))

# Plot mesh to check
#plot_mesh.plot_mesh(coord_array, tri_array)

# Calculate HKS
steps = 0.001
iters = 1000
HKS = generate_hks.generate_hks(coord_array, tri_array, steps, iters)
  
# Plot HKS if required at iter n
#n = [0, 10, 50, 299, 799]
#plot_hks.plot_hks(coord_array, tri_array, HKS, n)
  
# Calculate persistence
v = persistence.persistence(HKS)
np.save('temp/hks',v)

f_name = ''

# # Plot persistence value
plot_persistence.plot_persistence(coord_array, tri_array, v, 'value')
#plot_persistence.plot_persistence(coord_array, tri_array, v, 'level')
  
# get the connection matrix of points, and save in file 'connection_matrix_points.npy'
adjpts = connection_matrix_point.connection_matrix(coord_array,tri_array)

save_path = '/Users/ting/Downloads/clusters/through_hole_blind_hole/'
save_name = feature2 + '_%d'%fname
np.savez_compressed(save_path+save_name, persistence = v,adjpts=adjpts)

# Find clusters
# simil = [0.7, 0.75, 0.8, 0.85, 0.9]  # Similarity percentages
# clusters = cluster_old.cluster(coord_array, tri_array, v, simil)
# 
# #cluster_adjmap = cluster.get_attributed_cluster_adj_matrix(simil,clusters,tri_array)
# #plt.pcolor(cluster_adjmap, edgecolors='k', linewidths=1)
# # plt.xticks(range(60))
# # plt.yticks(range(60))
# # plt.show()
# 
# # Plot found clusters
# plot_clusters.plot_clusters(coord_array, tri_array, clusters, simil)

