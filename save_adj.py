# -*- coding: utf-8 -*-
import numpy as np

# def get_feauture_cluster_adjm(pts,adjm,cluster): #p27版，无法读取模型,改 pts
#     
#     f_clusters = np.unique(cluster[-2,:][pts]).astype(np.int)-1
#     
#     new_adjm = adjm[f_clusters,:][:,f_clusters]
#     return new_adjm

def get_feauture_cluster_adjm(model,adjm,cluster):
    
    faces = model.features[0].faces 
    
    pts = np.array([])
    for fc in faces:
        pts = np.append(pts,model.faces[fc].mesh_points) #numpy array
        
    pts = np.unique(pts).astype(np.int)

    f_clusters = np.unique(cluster[0,:][pts]).astype(np.int)-1
    
    new_adjm = adjm[f_clusters,:][:,f_clusters]
    return new_adjm

def get_feauture_cluster_and_adjm(feature_model,my_model,adjm,cluster):
    
    if len(feature_model.features) != 2:
        raise NameError('Feature 数量%d有错，请检查!'% len(feature_model.features))
    
    
    adjms = []; clusters = []
    for fid,feature in enumerate(feature_model.features):
        
        faces = []
        for face_key in feature.faces:
            if  type(feature_model.faces[face_key].ID) == list:
                faces.extend(feature_model.faces[face_key].ID)
            else:
                faces.append(feature_model.faces[face_key].ID)
        
        pts = np.array([])
        for fc in faces:
            pts = np.append(pts,my_model.faces[fc].mesh_points) #numpy array
            
        pts = np.unique(pts).astype(np.int)
    
        f_clusters = np.unique(cluster[0,:][pts]).astype(np.int)-1
        
        new_adjm = adjm[f_clusters,:][:,f_clusters]
        
        fcluster = np.zeros((2,f_clusters.size))
        
        for i,c in enumerate(f_clusters):
            pts_indx = np.where(cluster[0,:] == c+1)
            pts_in_feature = np.intersect1d(pts,pts_indx[0])   #some points in cluster but not in the feature    
            mean_value = np.mean(cluster[1,:][pts_in_feature])
            percentage = pts_in_feature.size /pts.size
            fcluster[0,i] = percentage
            fcluster[1,i] = mean_value
        adjms.append(new_adjm)
        clusters.append(fcluster)
        
    return adjms,clusters


def save_txt(fname,new_adjm,save_path):
    rename = fname[:-4]    
    size = new_adjm.shape[0]	
    
    for i in range(size):
        format = ''
        for j in range(size):
            if i==j:
                format += '%.11f '
            else:
                format += '%d '
        #with open('/data/userdata/yshi/adjm3/%s.txt'%rename,'ab') as f:
        with open(save_path+'%s.txt'%rename,'ab') as f:
            np.savetxt(f,new_adjm[i,:].reshape((1,size)),fmt=format)
        f.close()
