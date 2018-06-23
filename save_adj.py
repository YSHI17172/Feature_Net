import numpy as np

def get_feauture_cluster_adjm(model,adjm,cluster):
    
    faces = model.features[0].faces 
    
    pts = np.array([])
    for fc in faces:
        pts = np.append(pts,model.faces[fc].mesh_points) #numpy array
        
    pts = np.unique(pts).astype(np.int)

    f_clusters = np.unique(cluster[-2,:][pts]).astype(np.int)-1
    
    new_adjm = adjm[f_clusters,:][:,f_clusters]
    return new_adjm
    

def save_txt(fname,ftype,new_adjm):
 
    tname = str(ftype) + fname[-10:-4]
    
    size = new_adjm.shape[0]	
    
    clear = open('adjm/%s.txt'%tname,'w')
    clear.close()
    
    for i in range(size):
        format = ''
        for j in range(size):
            if i==j:
                format += '%.11f '
            else:
                format += '%d '
        with open('adjm/%s.txt'%tname,'ab') as f:
            np.savetxt(f,new_adjm[i,:].reshape((1,size)),fmt=format)
        f.close()
