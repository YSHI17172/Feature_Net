from cluster5 import *
import numpy as np
import datetime 

def cluster(coord_array, tri_array, v, simil):
    """ Function to find clusters of points with similar heat persistence values
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param simil: cluster similarity fractions
    :param v: (#points x 2) array with cluster points for part specified
    :return clusters: array of shape=(2*len(simil), #points) with cluster index/persistence values on even/odd rows
    """
    
    print("Finding clusters..")

    starttime = datetime.datetime.now()  

    newcluster = cluster5(coord_array, tri_array, simil, v)
    newcluster_i = newcluster[0,:]
    newcluster_v = newcluster[1,:]
        
    clmat = get_cluster_adj_matrix(newcluster_i, tri_array)
    
    # find very small clusters
    for tcli in range(int(np.amax(newcluster_i))):
        if np.count_nonzero(newcluster_i == tcli+1) < 9:
            # combine small clusters to smallest cluster of their neighbors
            neis_tcl = np.nonzero(clmat[:, tcli])[0]
            count = 0
            if np.size(neis_tcl) > 0 :
                for nei in neis_tcl: # find biggest neighbor cluster
                    nnei = np.count_nonzero(newcluster_i == nei+1)
                    
                    if nnei > count:
                        com_nei = nei
                        count = nnei
                pts_nei = np.nonzero(newcluster_i == com_nei+1)[0] # index of points in chosen cluster
                v_nei = np.amax(newcluster_v[pts_nei])                # value of points
                newcluster_v[newcluster_i == tcli+1] = v_nei
                newcluster_i[newcluster_i == tcli+1] = com_nei + 1 
    
    # resort the index of clusters, remove empty cluster
    while len(np.unique(newcluster_i)) != np.amax(newcluster_i):
        for i in range(1, int(np.amax(newcluster_i))+1):
            if i not in newcluster_i:
                for n in range(len(newcluster_i)):
                    if newcluster_i[n] > i:
                        newcluster_i[n] -= 1
                
    newcluster[0,:] = newcluster_i
    newcluster[1,:] = newcluster_v   
    endtime = datetime.datetime.now()  
    elapsedtime = (endtime - starttime).seconds
    print ("%d%% similarity complete. %d clusters found. Elapsed time is %s seconds."%(simil*100, np.amax(newcluster_i), elapsedtime))
    
    np.savez_compressed('temp/clusters', clusters=newcluster)
    print("Clusters complete.\n")

    return newcluster


def get_cluster_adj_matrix(cluster, tri):  # Check filter_clusters has duplicate function (name&code)

    conn = np.hstack([np.vstack([tri[:, 0], tri[:, 1]]), np.vstack([tri[:, 1], tri[:, 2]]),
                          np.vstack([tri[:, 2], tri[:, 0]])]).conj().T.astype(int)

    clusadj = np.zeros(shape=(3*tri.shape[0], 2), dtype=int)
    """
    for k in range(cluster.shape[0]):
        inds1 = np.nonzero(conn[:, 0] == k)[0]
        inds2 = np.nonzero(conn[:, 1] == k)[0]
        clusno = cluster[k]
        clusadj[inds1, 0] = clusno
        clusadj[inds2, 1] = clusno
    """
    clusadj[:, 0] = cluster[conn[:, 0]]
    clusadj[:, 1] = cluster[conn[:, 1]]
    conn = 0 # clear momery
    adjclm = np.zeros(shape=(int(np.amax(cluster)), int(np.amax(cluster))), dtype=np.int8)
    for l in range(clusadj.shape[0]):
        a = clusadj[l, 0] - 1   # Python index, so shift down by 1
        b = clusadj[l, 1] - 1   # Python index, so shift down by 1

        if ((adjclm[a, b] == 0) and (adjclm[b, a]==0) and (a!=b)):
            adjclm[a, b] = 1
            adjclm[b, a] = 1

    return adjclm

def get_attributed_cluster_adj_matrix(simil,cluster,tri):  # Check filter_clusters has duplicate function (name&code)

    conn = np.hstack([np.vstack([tri[:, 0], tri[:, 1]]), np.vstack([tri[:, 1], tri[:, 2]]),
                          np.vstack([tri[:, 2], tri[:, 0]])]).conj().T.astype(int)
    
    clus = cluster[0,:]
    clusadj = np.zeros(shape=(3*tri.shape[0], 2), dtype=int)
    clusadj[:, 0] = clus[conn[:, 0]]
    clusadj[:, 1] = clus[conn[:, 1]]
    adjclm = np.zeros(shape=(int(np.amax(clus)), int(np.amax(clus))))
    for l in range(clusadj.shape[0]):
        a = clusadj[l, 0] - 1   # Python index, so shift down by 1
        b = clusadj[l, 1] - 1   # Python index, so shift down by 1

        if ((adjclm[a, b] == 0) and (adjclm[b, a]==0) and (a!=b)):
            adjclm[a, b] = 1
            adjclm[b, a] = 1
    for c in range(int(np.amax(clus))):
        pinc = np.where(clus == c+1)
        mean_value = np.mean(cluster[1,:][pinc[0]])
        adjclm[c,c] = mean_value

    return adjclm