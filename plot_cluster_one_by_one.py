# -*- coding: utf-8 -*-
import numpy as np
from mayavi import mlab


def plot_clusters(coord_array, tri_array, clusters,simn = 4):
    """ plot_clusters plots clusters with same persistence similarity levels.
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param clusters: array of shape=(2*len(simil), #points) with cluster index/persistence values on even/odd rows
    :param simil: array with %similarity values, shape=(x,)
    :return:
    """
    
    cluster_list = np.unique(clusters[simn*2,:]) # clusters 总数 

    # Create cluster plots for each similarity level
    for c in cluster_list:
        
        ci = np.copy(clusters[simn*2, :]) #复制编号栏
        cv = np.copy(clusters[simn*2+1, :]) * (ci == c) #copy row of persistence values, 非目标编号取0
        
        mlab.figure(figure="Cluster%d"%c, bgcolor = (1,1,1), fgcolor = (0,0,0),size = (2000,1000))
        
        # Create black wireframe mesh
        wire_mesh = mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], tri_array, scalars=None,
                                    line_width=0.1, representation='wireframe', color=(0, 0, 0), opacity=0.05)
        
        # Create face coloring
        wire_mesh.mlab_source.dataset.cell_data
        wire_mesh.mlab_source.dataset.cell_data.scalars = cluster_cell_values(tri_array, cv)
        wire_mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
        wire_mesh.mlab_source.update() 
        

        mesh = mlab.pipeline.set_active_attribute(wire_mesh, cell_scalars='Cell data')
        surf = mlab.pipeline.surface(mesh) 

        colors = np.array([[1, 1, 1, 0], [75, 173, 137, 255]])
        
        # Now use this colormap and show result
        surf.module_manager.scalar_lut_manager.lut.table = colors
        
       # Show plot
        mlab.show()


def cluster_cell_values(tri, cell_point_values):
    """ Function to select face values from lowest value corner points
    :param tri: vertex connection indices of the part triangular mesh
    :param cell_point_values: mesh triangle point values
    :return: mesh clus_face_val minimum values triangles
    """
    clus_face_val = np.zeros(tri.shape[0])
    for i in range(tri.shape[0]):
        alpha_count = np.count_nonzero(cell_point_values[tri[i, :]])
    
        if alpha_count > 0:
            # Vertices values triangle all nonzero, set face value to one
            clus_face_val[i] = 1
        else:
            # One or more triangle vertex point value is zero, set face value to zero
            clus_face_val[i] = 0

    return clus_face_val
    
    
if __name__ == "__main__":
    data = np.load('temp/coo.npz')
    clusters = np.load('temp/clusters.npz')['clusters']
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    data.close() 
    plot_clusters(coord_array, tri_array, clusters)
