import numpy as np
from scipy.sparse import load_npz

def cluster5(coord_array, tri_array, simil, v):
    """ Function to find clusters of points with similar heat persistence values
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param simil: array with cluster similarity fractions, shape=(#simil-percentages, )
    :param v: heat persistence level and value, array of shape (#points, 2)
    :return clusters: array of shape=(2*len(simil), #points) with cluster index/persistence values on even/odd rows
    """

    #notri = tri_array.shape[0]
    nopts = coord_array.shape[0]
    neighbor_similarity = simil

    p1 =v[:,1]

    adjpts = load_npz('temp/connection_matrix_points.npz')
    adjpts = adjpts.toarray()
    
    i = p1.argsort()[::-1] # i is index of sort p1 ,from large value to small
    x = p1[i]             # x is sorted p1 from large to small
    clusters = np.zeros(shape=(2, coord_array.shape[0]))
    count = 1                  # Cluster ID number, start at 1 to match MatLab
    visited = np.ravel(np.zeros(shape = (1, nopts)))
    step = 10

    while (i.size != 0): # loop over all points in value array
        if abs( i.size / float(nopts)) <= step/10.:
            print ("%5.2f%% points finished clustering."%(100-abs(i.size / float(nopts)*100)))
            step -= 1
        
        cl_list = []  
        cl_list.append(i[0]) # append first index, point index of the biggest value

        rootval = x[0]   # get the biggest value

        while (len(cl_list) != 0): 
            if (visited[cl_list[0]] == 0):
                if nopts < 100000:
                    adjpt = np.array(np.nonzero(adjpts[cl_list[0], :] == 1)) 
                else:
                    adjpt = adjpts[cl_list[0], :] - 1
                    remove = np.array([-1])
                    adjpt = np.setdiff1d(adjpt, remove)  #remove the empty postion in adjpt
                    
                adjpt = adjpt[np.nonzero(visited[adjpt] == 0)]   
            else:
                adjpt = []

            visited[cl_list[0]] = 1
            curpos = np.nonzero( i == cl_list[0])

            addpts = np.nonzero(p1[adjpt] > (neighbor_similarity*rootval))

            if (len(addpts[0]) != 0):
                list1 = cl_list
                list2 = (adjpt[addpts].T).tolist()
                cl_list = list1 + list2

            clusters[0,cl_list[0]] = int(count)
            clusters[1,cl_list[0]] = rootval
            cl_list.pop(0)  #deleting the first element

            i = np.delete(i,curpos)
            x = np.delete(x,curpos)
            if (len(cl_list) == 0):
                count += 1

    return clusters
