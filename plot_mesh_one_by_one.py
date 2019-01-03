from mayavi import mlab
import numpy as np

def plot_mesh(coord_array, tri_array):
    """ plot_mesh plots an unsorted triangular part mesh from xyz coordinates and triangle connectivity
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :return:
    """
    label = np.load('temp\label.npz')['label']   
    nl = np.amax(label).astype(int)
 
    for l in range(nl):
        plnsrf = np.zeros(shape=(np.amax(tri_array)+1, 1), dtype=int)
        
        vtx_in_label = np.unique(tri_array[np.nonzero(label == l+1)[0], :])              
        plnsrf[vtx_in_label] = 1
        
        if vtx_in_label.size > 3:
           
            mlab.figure(figure="Label %d"%(l+1), bgcolor = (1,1,1), fgcolor = (0,0,0))
        
            # Create black wireframe mesh
            wire_mesh = mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], tri_array,
                                            scalars=None, line_width=0.1, representation='wireframe', color=(0, 0, 0), opacity=0.05)
        
            # Set face coloring
            wire_mesh.mlab_source.dataset.cell_data
            wire_mesh.mlab_source.dataset.cell_data.scalars = planar_surf_cell_values(tri_array, plnsrf)
            wire_mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
            wire_mesh.mlab_source.update() 
        
            # point_data = wire_mesh.mlab_source.dataset.point_data
            # point_data.scalars = plnsrf #planar_surf_cell_values(tri_array, plnsrf)
            # point_data.scalars.name = 'Point data'
            # point_data.update()
        
            # Plot triangular mesh with face coloring
            mesh = mlab.pipeline.set_active_attribute(wire_mesh, cell_scalars='Cell data')
            surf = mlab.pipeline.surface(mesh)
        
            # Create colormap for the surf object. (2x4)
            # this is an array of (R, G, B, A) values (each in range 0-255)
            # First row white color with full transparency to mask faces with 0 value
            # Second row green MatLab R2015a color, fully opaque
            # if all values of face color is 0 or 1, the plot color is colors[0], so I identify the face value first.
            if len(np.unique(planar_surf_cell_values(tri_array, plnsrf))) == 1:
                colors = np.array([[75, 173, 137, 255], [75, 173, 137, 255]])
            else:
                colors = np.array([[1, 1, 1, 0], [75, 173, 137, 255]])
            
                # Now use this colormap and show result
            surf.module_manager.scalar_lut_manager.lut.table = colors
           
            #display the number of planes

            center = np.mean(coord_array[vtx_in_label],axis = 0)
                
            mlab.text(center[0], center[1], '%d'%(l+1), z=center[2],
                                width=0.05,)
            
            mlab.show()

def planar_surf_cell_values(triangles, cell_point_values):
    """ Function to determine face values from cell vertex values
    :param triangles: array with vertex connection indices of the part triangular mesh
    :param cell_point_values: array with mesh triangle point values
    :return plnsrf_face_val: array with values equal to 1 if all vertex points triangle equal 1, else 0
    """
    plnsrf_face_val = np.zeros(triangles.shape[0])
    for i in range(triangles.shape[0]):
        alpha_count = np.count_nonzero(cell_point_values[triangles[i, :]])
        if alpha_count > 0:
            # Vertices values triangle all nonzero, set face value to one
            plnsrf_face_val[i] = 1
        else:
            # One or more triangle vertex point value is zero, set face value to zero
            plnsrf_face_val[i] = 0

    return plnsrf_face_val

if __name__ == '__main__':
    data = np.load('temp/coo.npz'); #normv =  np.load('temp\pln_nv.npz')['normv']
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    plot_mesh(coord_array, tri_array)