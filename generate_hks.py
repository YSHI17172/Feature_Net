from scipy.sparse import csc_matrix, coo_matrix, linalg, spdiags
from math import pi
from laplacian import *
from normalized import *
import numpy as np
import timeit


def generate_hks(coord_array, tri_array, steps=0.001, iters=1000):
    """ generate_hks calculates the Heat Kernel Signature matrix.
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param steps: time step value in seconds, if steps & iters not given defaults to steps=0.001
    :param iters: integer number of time steps that are calculated, if steps & iters not given defaults to iters=1000
    :return HKS: point heat values per time step, array of shape=(#points, #iterations)
    """

    print("Calculating HKS..")

    L, m = laplacian(coord_array, tri_array)

    # Add tiny value to L diagonal to help solver
    Lcor = spdiags(1e-10*np.ones(L.shape[0]), 0, L.shape[0], L.shape[0])
    Lnew = L + Lcor
    
    if m.shape[0] < 300 :
        eigno = m.shape[0]-1 # in case of no enough mesh triangles
    else:
        eigno = 300
    print("Start eigenvalue calculations")
    tic = timeit.default_timer()
    D, V = linalg.eigsh(m, k=eigno, M=Lnew, sigma=eigno, return_eigenvectors=True, which='LM', mode='buckling')
    toc = timeit.default_timer()
    print("Eigenvalue caculation complete. Elapsed time = %s s" %(toc - tic))  # elapsed time in seconds

    i1 = tri_array[:, 0]
    i2 = tri_array[:, 1]
    i3 = tri_array[:, 2]
    v1 = coord_array[i3, :] - coord_array[i2, :]
    v2 = coord_array[i1, :] - coord_array[i3, :]
    # v3 = coord_array[i2, :] - coord_array[i1, :]

    n = np.cross(v1, v2)
    dblA = np.transpose(np.sqrt(np.sum(np.power(np.transpose(n), 2), axis=0)))
    total_area = np.sum(dblA)/2

    # Sort D and V by max absolute value first
    sort_ind = np.argsort(np.abs(D))[::-1]  # Reverse sort indices because abs largest value must come first
    D = D[sort_ind]
    V = V[:, sort_ind]

    # Scale D and normalize V
    D = np.real(D)
    D = (total_area/4/pi)*np.ones(D.shape)/D
    V = normalized(np.real(V), 0)

    # Start iterations for HKS generation
    t = steps   # start time rqual to step time
    t_start_HKS = timeit.default_timer()
    print("Start iterating for HKS calculations")

    HKS = np.zeros([V.shape[0], iters])
    for i in range(iters):
        # Record at what time first iteration begins
        if i == 0:
            t_iter = t_start_HKS

        # Display what is going on every 25 iters
        if np.remainder(i+1, 100) == 0:
            t_elapsed = timeit.default_timer() - t_iter
            print ("HKS iteration %s elapsed time %s s" %(i+1, t_elapsed))
            # Record at what time new batch of 100 iterations begins
            t_iter = timeit.default_timer()

        # Calculate HKS values
        H = np.dot((V**2), np.exp(np.dot(D, t)))
        HKS[:, i] = H

        t += steps

        if i == iters-1:
            toc = timeit.default_timer()
            t_elapsed = t_start_HKS - toc
            print("Last HKS iteration completed.\n")

    return HKS

    
if __name__ == "__main__":
    data = np.load('temp/coo.npz')  
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    data.close()
    HKS = generate_hks(coord_array, tri_array, steps=0.001, iters=1000)
    