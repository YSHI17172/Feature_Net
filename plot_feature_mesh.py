# -*- coding: utf-8 -*-
"""
只 plot 指定面mesh,PPT用
"""
import numpy as np
import plot_mesh
import mayavi
from read_dat import read_dat
from read_from_gmsh import read_gmsh

fname = 'sipe_socket_1.hh_1.dat'

path = 'test_case/mesh/%s'%fname
label_path = 'test_case/label/%s_label.npz'%fname
save_path = '/Users/User/OneDrive - University of South Carolina/Thesis/PICS/test_case/'

if 'dat' in fname:
    coord_array, tri_array = read_dat(path)
else:
    coord_array, tri_array = read_gmsh(path)
label = np.load(label_path)['label']

data = {'0_simple_bracket2_1.hh.msh':{'STEP':[[3,14],[9,10]],'SLOT':[[5,6,7]],'THROUGH':[[1],[2]]},
'1_spacer_66_1.hh.msh':{'STEP':[[18,23],[19,24]],'THROUGH':[[25],[14],[13],[12],[11]],'BLIND':[[5,7],[1,8],[4,6],[3,10],[2,9]]},
'goodpart1_1.hh_0.5.dat':{'STEP':[[11,12,35],[18,19,36],[25,26,37],[3,29,34]],'SLOT':[[14,15,16],[21,22,23],[2,7,28],[8,9,31]],'THROUGH':[[32],[33],[38],[39],[40]]},
'sipe_socket_1.hh_1.dat':{'STEP':[[2,4],[3,5],[7,8],[6,17]],'SLOT':[[11,12,13,19,20,21],[22,24,30,18,28,29,23,25,14,26,27]],'THROUGH':[[31],[33],[34],[35],[36],[37],[38]]},
'DEMO04_1.hh_0.5.dat':{'STEP':[[15,20]],'POCKET':[[42,46,47,48,84,87],[43,44,45,49,85,86],[23,24,25,26,68,69,70,71,50],[19,28,29,30,31,32,33,34,35,36,37,38,39,72,73,74,75,76,77,78,79,80,81,82,83]],
'THROUGH':[[63],[64],[88],[89]],'BLIND':[[13,61],[8,57],[14,62],[7,56],[40,66],[1,54],[2,55],[41,67],[12,60],[27,65],[9,58],[5,10],[11,59],[3,4]]}}

features = data[fname]
for key in features:
    
    for i,faces in enumerate(features[key]):
    
        save_name = '%s_%d'%(key,i)
        
        points = [];face_tri=np.ones((0,3))
        
        points_in_face = [];tri_in_face = []
        nl = np.amax(label).astype(int)  
        for i in range(1, nl+1):         
            vtx_in_label = np.unique(tri_array[np.nonzero(label == i)[0], :]) # points index
            points_in_face.append(vtx_in_label)
            tri_in_face.append(tri_array[np.nonzero(label == i)[0], :])
        
        for face_id in faces:
            points.extend(points_in_face[face_id-1])
            face_tri = np.vstack((face_tri,tri_in_face[face_id-1]))
        points = np.unique(points)
        
        for i,p in enumerate(points):
            face_tri[face_tri==p] = i
        face_coo = coord_array[points]
        
        feature = plot_mesh.plot_mesh(face_coo, face_tri.astype(np.int),save_name)
        
        mayavi.mlab.savefig(save_path+'%s/%s.jpg'%(fname.split('.')[0],save_name),size = (1500,1250) ,figure =feature)
        
mayavi.mlab.close(all=True)
    