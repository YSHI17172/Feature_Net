import numpy as np
import laplacian
import flood_fill
import half_edge
import datetime 
"""
find curvature at every point -- if large than 0.01 (not plane) mark edges
use edges to cluster facets -- connect single facets using floodfill
if some facets still single merge them to their neighbor facet with closest norm
"""

def segment_by_curvature(coord_array, tri_array):
    """ Function to detect planar surfaces and label them, it will separate curved surfaces from planar surfaces.
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :return label: array with surface id numbers per triangle, shape=(#triangles, )
    """
    print ("Recognizing surfaces...")
    starttime = datetime.datetime.now()  
    
    # Build numpy structured array half edge for an efficient computation
    M = half_edge.half_edge(coord_array, tri_array)
    nf = M['facet'].shape[0]

    L, Mlapl = laplacian.laplacian(coord_array, tri_array)
    mean_curv = np.dot(L.todense(), coord_array)
    mean_curv = np.sqrt(np.sum(mean_curv**2, axis=1))
    thresh = 0.001
    if True:
    #for thresh in np.linspace(0.001,0.1,100): 
    
        bndry_vtx = np.array((mean_curv > thresh).astype(int))
        bndry_edge = np.logical_and(bndry_vtx[M['edge'][:, 0]], bndry_vtx[M['edge'][:, 1]]).astype(int)
        
        nei = [[] for _ in range(nf)]  # Instead of cell(nf,1), list of lists where each list can be set independently
        for i in range(nf): # loop over all facets
            fh = M['facet_child_halfedge'][i] # facet i starts from halfedge fh
            start_fh = fh
            
            pf = M['halfedge_parent_facet'][M['halfedge_opposite_halfedge'][fh]].tolist() # opposite half edge - its facet
            if pf == 0 or bndry_edge[int(M['halfedge_parent_edge'][fh])] == 1: # fisrt facet or fh is edge
                pf = [np.nan]  # Use list, assign list with np.nan value to later prevent appending to list
    
            if not np.isnan(pf[0]): # else
                nei[i].append(pf)
    
            fh = M['halfedge_next_halfedge'][fh]
            while fh != start_fh: # repeat for the other two edges
                pf = M['halfedge_parent_facet'][M['halfedge_opposite_halfedge'][fh]].tolist() 
                if pf == 0 or bndry_edge[int(M['halfedge_parent_edge'][fh])] == 1:
                    pf = [np.nan]
    
                if not np.isnan(pf[0]):
                    nei[i].append(pf)
                fh = M['halfedge_next_halfedge'][fh]
    
        label = np.arange(nf)  # Index shift used here
        old_label = np.zeros_like(label)  
    
        while np.sum(label != old_label): # cluster facets btween edges
    
            old_label = np.copy(label)
            label_map = np.arange(nf)  # Index shift
            for i in range(nf):
                list_with_lists = [[i]] + nei[i]
                flattened = [int(val) for sublist in list_with_lists for val in sublist]  # 'flatten' list_with_lists
                nei_label = label[flattened]
                min_nei_label = np.amin(label_map[nei_label])
                label_map[nei_label] = min_nei_label
    
            label = label_map[label]  # Shifted one down because later used as Python index
        
        #change label indexs from discrete to continuous
        un = np.unique(label[np.nonzero(label+1)[0]])  # Shift label when searching for non-zeros
        label_map[un] = np.arange(len(un))  # Index shift
        label = label_map[label]
    
        sorted_label = np.sort(label)
        unique_label, ia = np.unique(sorted_label, return_index=True)
        label_cnt = np.hstack((ia[np.arange(1, len(ia))], len(label)))-ia
        one_el_label = np.nonzero(label_cnt == 1)[0] # single labels
    
        label_map = np.ones_like(unique_label) 
        label_map[one_el_label] = 0
        bw_label = label_map[label]
    
        i = np.nonzero(bw_label == 0)[0]  # single facets 
        while len(i):  # so while not empty i
            mask = flood_fill.flood_fill(M, i[0], bw_label) #get all connected single facets
            if len(mask) > 2:
                label[mask] = np.amax(label)+1 # add new label number
    
            bw_label[mask] = 1 # give up real single facet
            i = np.nonzero(bw_label == 0)[0] # next loop
    
    
        un = np.unique(label[np.nonzero(label+1)[0]])
        label_map = np.zeros(un[-1]+1).astype(int)  # Create new label_map, Python cannot automatically resize var
        label_map[un] = np.arange(len(un))  # Index shift
        label = label_map[label]
    
        sorted_label = np.sort(label)
        unique_label, ia = np.unique(sorted_label, return_index=True)
        label_cnt = np.hstack((ia[np.arange(1, len(ia))], len(label)))-ia
        one_el_label = np.nonzero(label_cnt == 1)[0]
    
        # prestore the facet index, in case one point label is another one's target in next loop
        pts_one_label = [np.nonzero(label == one_el_label[j])[0] for j in range(len(one_el_label))]
    
        for j in range(len(one_el_label)):
            fid = np.nonzero(label == one_el_label[j])[0]
            fid_temp = fid[0]
            v1 = M['vertex'][M['facet'][fid_temp, 1], :] - M['vertex'][M['facet'][fid_temp, 0], :] # line10
            v2 = M['vertex'][M['facet'][fid_temp, 2], :] - M['vertex'][M['facet'][fid_temp, 0], :] # line20
            my_n_temp = np.cross(v1, v2)  # norm of facet
            my_n = my_n_temp/np.sqrt(np.sum(my_n_temp**2)) # norm vector
    
            fh = int(M['facet_child_halfedge'][pts_one_label[j]])# get the original index
            start_fh = fh
    
            pf = M['halfedge_parent_facet'][M['halfedge_opposite_halfedge'][fh]].astype(int)
            v1 = M['vertex'][M['facet'][pf, 1], :] - M['vertex'][M['facet'][pf, 0], :]
            v2 = M['vertex'][M['facet'][pf, 2], :] - M['vertex'][M['facet'][pf, 0], :]
            n_temp = np.cross(v1, v2)
            n = n_temp/np.sqrt(np.sum(n_temp**2)) # norm vector of nei facet
            nei_n = n
            nei_id = pf
            fh = int(M['halfedge_next_halfedge'][fh])  # next edge
    
            while fh != start_fh:
                pf = M['halfedge_parent_facet'][M['halfedge_opposite_halfedge'][fh]].astype(int)
                v1 = M['vertex'][M['facet'][pf, 1], :] - M['vertex'][M['facet'][pf, 0], :]
                v2 = M['vertex'][M['facet'][pf, 2], :] - M['vertex'][M['facet'][pf, 0], :]
                n_temp = np.cross(v1, v2)
                n = n_temp/np.sqrt(np.sum(n_temp**2))
                nei_n = np.vstack((nei_n, n))
                nei_id = np.hstack((nei_id, pf))
                fh = int(M['halfedge_next_halfedge'][fh])
    
            max_id = np.argmax(np.dot(nei_n, my_n.conj().T)) # get the smallest angle between nei norm and original facet
            label[fid] = label[nei_id[max_id]] # merge this label to the max one
    
        un = np.unique(label[np.nonzero(label)[0]])
        label_map[un] = np.arange(1, len(un)+1)
        label = label_map[label]  # Shifted id numbers w.r.t MATLAB
        
        endtime = datetime.datetime.now() 
        elapsedtime = (endtime - starttime).seconds
        print ("%d surfaces recognized. Elapsed time is %s seconds." %(np.amax(label)+1,elapsedtime))
        
        label+=1
        
        print(thresh,np.amax(label))
        
    np.savez_compressed('temp/label', label=label)
    return label  # Work with MATLAB label format
    
if __name__ == "__main__":
    data = np.load('temp/coo.npz')  
    tri_array = data['tri_array'] ; coord_array = data['coord_array']
    label = segment_by_curvature(coord_array, tri_array)
    