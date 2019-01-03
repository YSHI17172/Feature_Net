import numpy as np
import datetime 
import half_edge
import sys

def find_planar_surf_manually(coord_array, tri_array,label, test = None):
    """ Find part planar surfaces by manually select the planes don't need
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param r: # Threshold value for planar surface detection
    :return plnsrf: array of shape=(#points,) with binary values indicating vertex on planar surface, 1, or not, 0
    :label_pln: array of  shape=(#labels,) with binary values indicating label as planar surface or not
    """
    from plot_mesh_nos import plot_mesh
    
    
    nl = np.amax(label).astype(int)
    plnsrf = np.zeros(shape=(np.amax(tri_array)+1, 1), dtype=int)

    # Build numpy structured array half edge for an efficient computation
    M = half_edge.half_edge(coord_array, tri_array)
    
       # relate points to labels
    for i in range(1, nl+1):         
        vtx_in_label = np.unique(M['facet'][np.nonzero(label == i)[0], :]) # points index
        plnsrf[vtx_in_label] = i
    

    plot_mesh(coord_array, tri_array,plnsrf) 
            
    
def find_planar_surf(coord_array, tri_array):
    """ Find part planar surfaces
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param r: # Threshold value for planar surface detection
    :return plnsrf: array of shape=(#points,) with binary values indicating vertex on planar surface, 1, or not, 0
    :label_pln: array of  shape=(#labels,) with binary values indicating label as planar surface or not
    """
    #normv = norm(coord_array, tri_array)
    #plnsrf = filter_surf(coord_array, tri_array,normv)
    
    label = np.load('temp/label.npz')['label']
    nl = np.amax(label).astype(int)
    
    label_pln = np.zeros(nl)
    plnsrf = np.zeros(shape=(np.amax(tri_array)+1, 1), dtype=int)

    # Build numpy structured array half edge for an efficient computation
    M = half_edge.half_edge(coord_array, tri_array)
    
    for i in range(1, nl+1):         
        vtx_in_label = np.unique(M['facet'][np.nonzero(label == i)[0], :]) # points index
        pec = vtx_in_label.size/ float(np.amax(tri_array))
        label_pln[i-1] = pec
        plnsrf[vtx_in_label] = i
        
    pln_sort = np.sort(label_pln)
   
    
    print (pln_sort[-10:])
    
    # find id of specific surface
    copy = np.copy(label_pln)
    for i in range(30):
        print (np.argmax(copy),np.amax(copy))
        copy[np.argmax(copy)]=0
    
    
    r = pln_sort[-2] # preset a value, in case shape is simple   
    rn = 2   
    for si in range(2,nl):
        rate =pln_sort[-si-1] / pln_sort[-si]
        if rate < 0.7:
            r = pln_sort[-si]*0.99; rn = si
            break
    print ("Select the %d biggest as thresh %.3f." %(rn,r))
        
    #label_pln[[5,111,110]] = 1 # NIST
    #label_pln[[7,8,6,9]] = 1; label_pln[[0]] = 0.04# NIST RANDOM 1
    #label_pln[[280,266,5,281]] = 1 # NIST mirror
    #label_pln[[11,14,13,12]] = 1;label_pln[[233,232]] = 0 # NIST RANDOM 2
    #label_pln[[9,10,8,11]] = 1 # bridge
    #label_pln[[23,24,25,26]] =1 # multi section 02

    
    label_filtered = np.nonzero(label_pln>r)[0]+1
    for label in label_filtered:
        plnsrf[plnsrf==label] = 0
    
    plnsrf = ((plnsrf > 0)*1).flatten()
    label_plnar = ((label_pln < r) * 1).flatten()
    np.savez_compressed('temp/pln', plnsrf=plnsrf,label_pln = label_plnar)
    
    return plnsrf

def norm(coord_array, tri_array):
    
    print("Finding planar surfaces..")
   
    normv = np.zeros(shape=coord_array.shape)

    tritri_array = np.hstack([np.vstack([tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]]),
                              np.vstack([tri_array[:, 1], tri_array[:, 2], tri_array[:, 0]]),
                              np.vstack([tri_array[:, 2], tri_array[:, 0], tri_array[:, 1]])])
    tritri = tritri_array.conj().T
    
    st_norm = datetime.datetime.now() 
    for i in range(coord_array.shape[0]):
        trind = np.nonzero(tritri[:, 0] == i)[0]
        adjpt1 = tritri[trind, 1]
        adjpt2 = tritri[trind, 2]

        coord1 = coord_array[adjpt1, :]
        coord2 = coord_array[adjpt2, :]

        v1 = coord1 - np.ones(shape=(coord1.shape[0], 1))*coord_array[i, :]
        v2 = coord2 - np.ones(shape=(coord2.shape[0], 1))*coord_array[i, :]

        normvis = np.cross(v1, v2)
        normvi = np.sum(normvis, axis=0)
        normv[i, :] = normvi/np.sqrt(np.sum(normvi**2))
    et_norm = datetime.datetime.now()
    t_norm = (et_norm - st_norm).seconds
    print ("Normalization of %d points complete. Elapsed time is %s seconds."%(coord_array.shape[0],t_norm))
    np.savez_compressed('temp/pln_nv', normv=normv)  
    return normv
    

def filter_surf(coord_array, tri_array,normv):
    print ("Filtering furfaces..")
    plnsrf = np.zeros(shape=(np.amax(tri_array)+1, 1), dtype=int)
    st_plan = datetime.datetime.now()

    surfn = np.zeros(shape=(coord_array.shape[0], 1),dtype=int)
    pln_list = []
    count = 1  # Keep at 1 to match MatLab
    
    while (surfn == 0).any():
        i = np.nonzero(surfn == 0)[0][0]
        abc = normv[i, :]
        d = -np.sum(abc*coord_array[i, :])  # element-wise multiplication

        dist = np.abs(np.sum((np.ones(shape=(coord_array.shape[0], 1))*abc)*coord_array, axis=1)
                      + d*np.ones(shape=(coord_array.shape[0], 1)).ravel())  # Use ravel to avoid nxn shape from np.abs

        npts = np.nonzero(dist == 0)[0]  # Index, so starts at zero
        surfn[npts] = count
        pln_list.append(len(npts))

        count += 1
        
    pln = np.array(pln_list)/float(coord_array.shape[0])
    """
    # find id of specific surface
    copy = np.copy(pln)
    for i in range(20):
        print np.argmax(copy)
        copy[np.argmax(copy)]=0
    """

    et_plan = datetime.datetime.now()  
    t_plan = (et_plan - st_plan).seconds
    print ("%d planes separation complete. Elapsed time is %s seconds.\n")%(count-1,t_plan)
    
    pts = np.array([])
    
    # find the correct r by comparing area ratio change
    pln_sort = np.sort(pln)
    
    r = pln_sort[-2] # preset a value, in case shape is simple   
    rn = 2   
    for si in range(2,pln_sort.size):
        rate =pln_sort[-si-1] / pln_sort[-si]
        if rate < 0.7:
            r = pln_sort[-si]*0.99; rn = si
            break
    print("Select the %d biggest as thresh %.3f.")%(rn,r)
    
    # Identify simple shape with all large surfaces.
    if count <= 6:
        plns = np.nonzero(pln)[0]

    else:
        pln[23199] = 1
        
        plns = np.nonzero(pln < r)[0]
   
    for i in range(len(plns)):
        pts = np.hstack([pts, (np.nonzero(surfn == plns[i]+1)[0]).conj().T])

    plnsrf[pts.astype(int)] = 1
    
    plnsrf = plnsrf.flatten()
    
    np.savez_compressed('temp/pln', plnsrf=plnsrf)

     # code to reduce points by removing these big surface
    """   
    transi = np.zeros(np.shape(coord_array)[0],dtype=int) # transfer info, new point index
    
    # produce new cor and tri for less computation
    coord_array_reduced = np.copy(coord_array)
    tri_array_reduced = np.copy(tri_array)
    index = 0; ti = 0
    while index < np.shape(coord_array_reduced)[0]:
        # not in planar surface, or at the end of loop, the number of tri is uneven   
        total = np.shape(coord_array_reduced)[0] - 1
        if (plnsrf[index] == 0 and index < total) or (index == total and np.shape(tri_array_reduced)[0] % 2 == 1):
            coord_array_reduced = np.delete(coord_array_reduced, index, 0)
            for n in range(3): # delete relevant triangle
                trii = np.nonzero(tri_array_reduced[:,n] == index)[0]
                tri_array_reduced = np.delete(tri_array_reduced, trii, 0)
            tri_array_reduced[tri_array_reduced > index] -= 1
            plnsrf = np.delete(plnsrf,index)
            if index == total:
                index -= 1
                transi[ti] = 0
                ti -= 2

        else:
            index += 1
            transi[ti] = index # mark new index of point
        ti += 1
    return transi, coord_array_reduced, tri_array_reduced  
    """
    return plnsrf

if __name__ == '__main__':
    import plot_planar_surf
    data = np.load('temp/coo.npz'); #normv =  np.load('temp/pln_nv.npz')['normv']
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    #plan_surf = filter_surf(coord_array, tri_array,normv)
    #plan_surf = find_planar_surf(coord_array, tri_array)
    plan_surf = find_planar_surf_manually(coord_array, tri_array)
    plot_planar_surf.plot_planar_surf(coord_array, tri_array, plan_surf)