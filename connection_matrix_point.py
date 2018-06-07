import numpy as np
from scipy.sparse import coo_matrix,save_npz

def connection_matrix(coord_array, tri_array):
    # function to get the connection matrix of points
    # when points < 100000, get a n*n matrix, its very fast
    # connection matrix of pionts takes too much memory, for example, array 200,000 * 200,000 with dtype int 8 takes 74GB
    # (200,000*200,000* 2byte /(2**30)) = 74
    # so we cant build matrix points * points, we use points * 8, becasue the mesh is triangles in squares
    # so the max number of connect points is 8
    # when points > 100000, dont have enough memory, so have to loop over all points
    
    I = np.vstack((tri_array[:,0],tri_array[:,1],tri_array[:,2])).ravel()
    J = np.vstack((tri_array[:,1],tri_array[:,2],tri_array[:,0])).ravel()
    conn = np.column_stack((I,J))
    
    nopts = coord_array.shape[0]
    notri = tri_array.shape[0]
    
    if nopts < 100000:
        adjpts = coo_matrix((np.ravel(np.ones(shape = (3*notri, 1))),\
        (I.ravel(), J.ravel())),shape = (nopts,nopts),dtype=np.int8)
    
    else:
        adjpts = np.zeros(shape=(notri, 8),dtype=int)
        for i in range(nopts):                  # loop over all pair of points
            temp = []

            for adjp in conn[conn[:,0]==i][1]:
                if adjp not in temp:
                    temp.append(adjp+1)        # +1 convert index because of fixed shape 8
            for adjp in conn[conn[:,1]==i][0]:
                if adjp not in temp:
                    temp.append(adjp+1)
            temp.sort()
            for ii in range(len(temp)):
                adjpts[i,ii] = temp[ii]
    save_npz('temp/connection_matrix_points', adjpts)
    #print (adjpts)
    #return adjpts

if __name__ == "__main__":
    data = np.load("temp/coo.npz")
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    data.close()
    connection_matrix(coord_array, tri_array)
    