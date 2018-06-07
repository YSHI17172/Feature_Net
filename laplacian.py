import numpy as np
from scipy.sparse import csc_matrix, coo_matrix


def laplacian(coord_array, tri_array):
    """ laplacian implements the laplace beltrami operator discretization of a triangular part mesh.
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :return L, M: laplace beltrami discretization matrices of the HKS generalized eigenvalue problem M*x = lambda*L*x,
    L and M have shape(#points, #points)
    """

    i1 = tri_array[:, 0]
    i2 = tri_array[:, 1]
    i3 = tri_array[:, 2]

    v1 = coord_array[i3, :] - coord_array[i2, :]
    v2 = coord_array[i1, :] - coord_array[i3, :]
    v3 = coord_array[i2, :] - coord_array[i1, :]

    n = np.cross(v1, v2)
    dblA = np.transpose(np.sqrt(np.sum(np.power(np.transpose(n), 2), axis=0)))

    # cot12 = inner1d(v1,v2) # This is slower than einsum
    cot12 = -0.5*(np.einsum('ij,ij->i', v1, v2)/dblA)
    cot23 = -0.5*(np.einsum('ij,ij->i', v2, v3)/dblA)
    cot31 = -0.5*(np.einsum('ij,ij->i', v3, v1)/dblA)
    diag1 = -cot12-cot31
    diag2 = -cot12-cot23
    diag3 = -cot31-cot23

    # csc is faster than coo in calculations, nonzero elements that Python displays is not correct
    ind_iL = np.hstack((i1, i2, i2, i3, i3, i1, i1, i2, i3))
    ind_jL = np.hstack((i2, i1, i3, i2, i1, i3, i1, i2, i3))
    vL = np.hstack((cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3))
    L = csc_matrix(coo_matrix((vL, (ind_iL, ind_jL))))  # csc is faster than coo in calculations

    # Create a sparse csc_matrix m, nonzero elements that Python displays is not correct
    diag_v = dblA/6.
    ind_iM = np.hstack((i1, i2, i3))
    ind_jM = np.hstack((i1, i2, i3))
    vM = np.hstack((diag_v, diag_v, diag_v))
    M = csc_matrix(coo_matrix((vM, (ind_iM, ind_jM))))  # csc is faster than coo in calculations

    return L, M
