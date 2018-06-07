from mayavi import mlab


def plot_hks(coord_array, tri_array, HKS, n):
    """ plot_hks plots values at index n of the heat kernel signature matrix of the triangular coord & tri mesh.
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param HKS: array of size (#points x #steps) with heat values per point per time step
    :param n: integer n to select desired HKS column at time t for plotting
    :return:
    """

    # Loop over values in n to create HKS plots at different time steps
    for i in n:
        mlab.figure(figure= str(i) + "_steps_HKS" , bgcolor = (1,1,1), fgcolor = (0,0,0))
        
        # Create wireframe mesh
        mesh = mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], tri_array, scalars=None,
                                    line_width=0.1, representation='wireframe', color=(0, 0, 0), opacity=0.05)

        # Create face coloring
        point_data = mesh.mlab_source.dataset.point_data
        point_data.scalars = HKS[:, i]
        point_data.scalars.name = 'Point data'
        point_data.update()

        mesh2 = mlab.pipeline.set_active_attribute(mesh, point_scalars='Point data')
        surf = mlab.pipeline.surface(mesh2)
        
        # Add colorbar
        mlab.colorbar(surf, orientation='vertical')
        
        # Show plot
        mlab.show()
