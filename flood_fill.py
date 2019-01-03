import numpy as np


def flood_fill(he_mesh, seed, label):
    """ Function to propagate a seed on a surface until boundaries are reached,
    used for surface detection by segment_by_curvature
    :param he_mesh: numpy structured array half edge structure
    :param seed: integer point index
    :param label: point labels, array with zeros and ones, shape=(#points,)
    :return mask: numpy array with python indices of points to be masked, shape=(x,)
    """

    M = he_mesh
    list = []
    list.append(seed)  # Convert to list for use in while check, can be done because seed is always int
    mask = []  # Switch to list instead of np.array and later switch back
    visited = np.zeros(M['facet'].shape[0],dtype = np.int8)
    while list:  # equivalent to ~isempty(list)
        curr = list[0]
        if visited[curr] == 1:
            list.pop(0)
            continue  # Keep this here?

        mask.append(curr)  # MATLAB creates a column vector
        visited[curr] = 1
        list.pop(0)

        fh = 0
        start_fh = M['facet_child_halfedge'][curr]
        while start_fh != fh:
            if fh == 0:
                fh = start_fh

            oh = M['halfedge_opposite_halfedge'][fh]
            nf = M['halfedge_parent_facet'][oh.flatten()[0]].astype(int)  # Might give index errors, use flatten() to solve
            if nf != 0 and label[nf] == 0 and visited[nf] == 0:
                list.append(nf)

            fh = M['halfedge_next_halfedge'][fh.flatten()]  # Prevent ndarray input

    return np.asarray(mask)
