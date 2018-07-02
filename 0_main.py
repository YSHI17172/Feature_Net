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

#part_name = "ISO/Boss2.dat"

data = np.load('input_mesh/slot/slot_345.npz')
tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
data.close()
#coord_array, tri_array = read_dat.read_dat(part_name)
print ("This part hase %d points, and %d triangles in mesh.\n"%(np.shape(coord_array)[0], np.shape(tri_array)[0]))

# Plot mesh to check
plot_mesh.plot_mesh(coord_array, tri_array)

# # Calculate HKS
# steps = 0.001
# iters = 1000
# HKS = generate_hks.generate_hks(coord_array, tri_array, steps, iters)
# 
# # Plot HKS if required at iter n
# #n = [0, 10, 50, 299, 799]
# #plot_hks.plot_hks(coord_array, tri_array, HKS, n)
# 
# # Calculate persistence
# v = persistence.persistence(HKS)
# 
# # # Plot persistence value
# plot_persistence.plot_persistence(coord_array, tri_array, v, 'value')
# plot_persistence.plot_persistence(coord_array, tri_array, v, 'level')
# 
# # get the connection matrix of points, and save in file 'connection_matrix_points.npy'
# connection_matrix_point.connection_matrix(coord_array,tri_array)
# 
# # Find clusters
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
# 
