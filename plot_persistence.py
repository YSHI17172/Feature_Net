from mayavi import mlab


def plot_persistence(coord_array, tri_array, v, option):
    """ plot_persistence_value plots the persistence level
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param v: array of shape=(#points, 2) with persistence levels k in v[:, 0] and persistence values s in v[:, 1]
    param option: string with 'level', or 'value'
    :return:
    """
    
    mlab.figure(figure= "Persistence_%s"%option , bgcolor = (1,1,1), fgcolor = (0,0,0))

    # Create wireframe mesh
    wire_mesh = mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], tri_array, scalars=None,
                                line_width=0.1, representation='wireframe', color = (0, 0, 0), opacity=0.05)

    # Create face coloring
    wire_mesh.mlab_source.dataset.point_data
    if option == 'level':
        wire_mesh.mlab_source.dataset.point_data.scalars = v[:, 0]
    else: # option == 'value'
        wire_mesh.mlab_source.dataset.point_data.scalars = v[:, 1]

    wire_mesh.mlab_source.dataset.point_data.scalars.name = 'Point data'
    wire_mesh.mlab_source.update() 

    mesh = mlab.pipeline.set_active_attribute(wire_mesh, point_scalars='Point data')
    surf = mlab.pipeline.surface(mesh)

    # Add colorbar
    mlab.colorbar(surf, orientation='vertical')

    # Show plot
    mlab.show()
