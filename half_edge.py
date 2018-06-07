import numpy as np


def half_edge(coord_array, tri_array):
    """ Creates a new mesh data structure in half edge format which is very efficient for finding neighbouring points.
    :param coord_array: XYZ mesh coordinates in numpy array, shape=(#vertices, 3)
    :param tri_array: mesh vertex connectivity numpy array, shape=(#faces, 3)
    :return M:  Dictionary M with half edge data structure fields
                M['vertex']: coord_array
                M['vertex_incoming_halfedge']: numpy array
                M['vertex_outgoing_halfedge']: numpy array
                M['halfedge']: [tri*3,2(points?)]
                M['halfedge_next_halfedge']: [tri*3,1] another halfedge in same facet \[1,2,0]
                M['halfedge_prev_halfedge']: [tri*3,1] another halfedge in same facet \[2,0,1]
                M['halfedge_opposite_halfedge']: numpy array
                M['halfedge_parent_edge']: numpy array
                M['halfedge_parent_facet']: [tri*3,1] halfedge to tri?
                M['edge']: numpy array  every line?
                M['edge_child_halfedge']: numpy array
                M['facet']: tri_array
                M['facet_child_halfedge']: [tri,1]/[0,3,6,9...] facet inclues 3 half edges
                M['boundary_halfedge_idx']: numpy array
    """

    # Load coord and tri variables
    F = np.copy(tri_array)
    nf = tri_array.shape[0]
    
    # Make a list of non-boundary halfedge + next/prev information, all this block is in Python index format
    halfedge_data = np.reshape(
        np.reshape((np.column_stack((F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2], F[:, 0])).conj().T),
                   (6*nf, 1), order='F'), (2, 3*nf), order='F').conj().T
    temp_halfedge_nextprev = np.arange(halfedge_data.shape[0])
    temp_halfedge_nextprev = np.reshape(temp_halfedge_nextprev, (3, int(len(temp_halfedge_nextprev)/3)), order='F')
    halfedge_next_halfedge = np.vstack((temp_halfedge_nextprev[1, :], temp_halfedge_nextprev[2, :],
                                        temp_halfedge_nextprev[0, :]))
    halfedge_next_halfedge = halfedge_next_halfedge.ravel(order='F')
    halfedge_prev_halfedge = np.vstack((temp_halfedge_nextprev[2, :], temp_halfedge_nextprev[0, :],
                                        temp_halfedge_nextprev[1, :]))
    halfedge_prev_halfedge = halfedge_prev_halfedge.ravel(order='F')
    facet_child_halfedge = np.arange(nf).conj().T*3  # Python index format
    
    # Link them to their parent facet
    halfedge_parent_facet = np.reshape(np.tile(np.arange(nf), (3, 1)), (3*nf, 1), order='F')
    
     # Make a list of edges. Keep track on the connection to each halfedge.
    halfedge_order = (halfedge_data[:, 0] < halfedge_data[:, 1]).astype(int)
    # Swtich Vextex index S-L
    temp_edge = np.column_stack((halfedge_order*halfedge_data[:, 0]+((halfedge_order == 0).astype(int))*halfedge_data[:, 1],
                           halfedge_order*halfedge_data[:, 1]+((halfedge_order == 0).astype(int))*halfedge_data[:, 0]))
    temp_edge_child_halfedge = np.arange(1, temp_edge.shape[0]+1).conj().T
    edge, ia, ic = unique_rows(temp_edge)  # Use custom function to return unique rows and indices thereof
    sorted_ic = np.sort(ic)
    iic_ls = np.argsort(ic)  # MatLab puts smalles ind first, Python the largest index first of every pair in sorted_ic
    if iic_ls.size % 2 != 0: # in case the number of triangle is odd
        odd_temp = iic_ls[-1]
        iic_ls = iic_ls[:-1]
    iic_reshape_sl = np.sort(np.reshape(iic_ls, (-1, 2)), axis=1)  # Reshape in nx2 with sorted rows small to large
    iic = iic_reshape_sl.ravel()  # Ravel (flatten) to 1D array, now same result as in MatLab
    if 'odd_temp' in locals(): # in case the number of triangle is odd
        iic = np.append(iic,odd_temp)
    halfedge_parent_edge = ic

    ne = edge.shape[0]
    
    # Link halfedges to their parent edges. While doing so, create boundary
    # halfedges if there is only one halfedge correspond to a given edge.
    edge_child_halfedge = np.zeros(len(edge), dtype=int)
    edge_child_halfedge[sorted_ic] = temp_edge_child_halfedge[iic]
    boundary_halfedge = 0
    i = 1  # Index shift
    
    while True:  # Test loop for other parts as well
        if sorted_ic[i] != (i-1)/2:  # Index shift i-1
            sorted_ic = np.hstack((sorted_ic[0:i], (i-1)/2, sorted_ic[i:int(len(sorted_ic))+1]))
            halfedge_data = np.vstack((halfedge_data, np.array([halfedge_data[edge_child_halfedge[int((i-1)/2)], 1],
                                                                halfedge_data[edge_child_halfedge[int((i-1)/2)], 0]])))
            halfedge_parent_facet = np.append(halfedge_parent_facet, np.zeros((1,1)), axis=0)
            halfedge_next_halfedge = np.append(halfedge_next_halfedge, 0)
            halfedge_prev_halfedge = np.append(halfedge_prev_halfedge, 0)
            iic = np.hstack((iic[0:i], len(halfedge_data)-1, iic[i:len(iic)+1]))
            halfedge_parent_edge = np.append(halfedge_parent_edge, (i-1)/2)
            if boundary_halfedge == 0:
                boundary_halfedge = len(halfedge_data)

        i += 2
        if i >= len(sorted_ic):
            break

    if np.mod(len(sorted_ic), 2):
        i = len(sorted_ic)
        halfedge_data = np.vstack((halfedge_data, np.array([halfedge_data[edge_child_halfedge[int((i-1)/2)], 1],
                                                            halfedge_data[edge_child_halfedge[int((i-1)/2)], 0]])))
        halfedge_parent_facet = np.append(halfedge_parent_facet, np.zeros((1,1)), axis=0)
        halfedge_next_halfedge = np.append(halfedge_next_halfedge, 0)
        halfedge_prev_halfedge = np.append(halfedge_prev_halfedge, 0)
        iic = np.hstack((iic[0:i], len(halfedge_data)-1))
        halfedge_parent_edge = np.append(halfedge_parent_edge, (i-1)/2)
        if boundary_halfedge == 0:
            boundary_halfedge = len(halfedge_data)

    edge_child_halfedge = np.reshape(iic, (ne, 2))

    # Link opposite halfedges
    halfedge_opposite_halfedge = np.zeros(edge_child_halfedge.shape[0]*2, dtype=int)
    halfedge_opposite_halfedge[edge_child_halfedge[:, 0]] = edge_child_halfedge[:, 1].flatten()
    halfedge_opposite_halfedge[edge_child_halfedge[:, 1]] = edge_child_halfedge[:, 0].flatten()
    halfedge_opposite_halfedge = halfedge_opposite_halfedge.conj().T
    nh = halfedge_data.shape[0]

    # Fill out next/prev relations for boundary halfedges
    if boundary_halfedge != 0:
        for j in range(boundary_halfedge, nh):
            next_bnd = halfedge_opposite_halfedge[halfedge_prev_halfedge[halfedge_opposite_halfedge[j]]]
            next_prev = halfedge_prev_halfedge[next_bnd]
            while next_prev != 0:
                next_bnd = halfedge_opposite_halfedge[halfedge_prev_halfedge[next_bnd]]
                next_prev = halfedge_prev_halfedge[next_bnd]

            halfedge_next_halfedge[j] = next_bnd
            halfedge_prev_halfedge[next_bnd] = j

    temp_unique_halfedge, vertex_outgoing_halfedge = np.unique(halfedge_data[:, 0], return_index=True)
    temp_unique_halfedge, vertex_incoming_halfedge = np.unique(halfedge_data[:, 1], return_index=True)

    # Create a Python dictionary to replace the MatLab struct
    M = {}
    M['vertex'] = coord_array
    M['vertex_incoming_halfedge'] = vertex_incoming_halfedge
    M['vertex_outgoing_halfedge'] = vertex_outgoing_halfedge
    M['halfedge'] = halfedge_data
    M['halfedge_next_halfedge'] = halfedge_next_halfedge
    M['halfedge_prev_halfedge'] = halfedge_prev_halfedge
    M['halfedge_opposite_halfedge'] = halfedge_opposite_halfedge
    M['halfedge_parent_edge'] = halfedge_parent_edge
    M['halfedge_parent_facet'] = halfedge_parent_facet
    M['edge'] = edge
    M['edge_child_halfedge'] = edge_child_halfedge
    M['facet'] = tri_array
    M['facet_child_halfedge'] = facet_child_halfedge
    M['boundary_halfedge_idx'] = boundary_halfedge

    return M


def unique_rows(data):
    """
    Custom function to get ndarray unique rows, indices and inverse indices
    :param data: input numpy ndarray
    :return: ndarray with unique rows, indices array i_a, inverse indices array i_c
    """
    uniq, i_a, i_c = np.unique(data.view(data.dtype.descr * data.shape[1]), return_index=True, return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, data.shape[1]), i_a, i_c
    
if __name__ == "__main__":
    data = np.load('temp/coo.npz')  
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    data.close()
    M = half_edge(coord_array, tri_array)