from mayavi import mlab
import numpy as np

def plot_mesh(i,coord_array, tri_array):
    """ plot_mesh plots an unsorted triangular part mesh from xyz coordinates and triangle connectivity
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :return:
    """
    mlab.figure(figure='Mesh%d'%i, bgcolor = (1,1,1), fgcolor = (0,0,0))

    # Create black wireframe mesh
    wire_mesh = mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], tri_array, scalars=None,
                                     line_width=0.1, representation='wireframe', color=(0, 0, 0), opacity=0.1)

    # Create face coloring
    wire_mesh.mlab_source.dataset.cell_data
    wire_mesh.mlab_source.dataset.cell_data.scalars = np.ones(shape=(tri_array.shape[0]))
    wire_mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
    wire_mesh.mlab_source.update() 

    mesh = mlab.pipeline.set_active_attribute(wire_mesh, cell_scalars='Cell data')
    surf = mlab.pipeline.surface(mesh, colormap='jet')

    # Retrieve the LUT colormap of the surf object. (256x4)
    # this is an array of (R, G, B, A) values (each in range 0-255)
    lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()

    # Modify lut for single green color
    lut[:, 0] = np.ones((lut.shape[0]))*75
    lut[:, 1] = np.ones((lut.shape[0]))*173
    lut[:, 2] = np.ones((lut.shape[0]))*137
    lut[:, 3] = np.ones((lut.shape[0]))*255

    # Now use this colormap
    surf.module_manager.scalar_lut_manager.lut.table = lut

    # Show plot
    mlab.show()

if __name__ == '__main__':
    data = np.load('temp\coo.npz'); #normv =  np.load('temp\pln_nv.npz')['normv']
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    data.close()
    plot_mesh(coord_array, tri_array)