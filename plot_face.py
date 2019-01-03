import numpy as np
from mayavi import mlab

def plot_clusters(i,coord_array, tri_array, clusters):
    mlab.figure(figure= 'face%d'%i, bgcolor = (1,1,1), fgcolor = (0,0,0))

    # Create black wireframe mesh
    wire_mesh = mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], tri_array, scalars=None,
                                        line_width=0.1, representation='wireframe', color=(0, 0, 0), opacity=0.05)

    # Create face coloring
    wire_mesh.mlab_source.dataset.cell_data
    wire_mesh.mlab_source.dataset.cell_data.scalars = cluster_cell_values(tri_array, clusters)
    wire_mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
    wire_mesh.mlab_source.update() 
    

    mesh = mlab.pipeline.set_active_attribute(wire_mesh, cell_scalars='Cell data')
    surf = mlab.pipeline.surface(mesh) 

    # Add colorbar
    mlab.colorbar(surf, orientation='vertical')

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
        clus_face_val[i] = np.amin(cell_point_values[tri[i, :]])  # amin

    return clus_face_val

if __name__ == "__main__":
    data = np.load('temp/coo.npz')
    clusters = np.load('temp/clusters.npz')['clusters']
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    data.close()
    plot_clusters(coord_array, tri_array, clusters)