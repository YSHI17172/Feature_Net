# -*- coding: utf-8 -*-
import numpy as np

def read_gmsh(path):
    with open(path,'r') as infile:
        content=infile.read()#将所有stl数据读到content中, str
    content = content.splitlines() #split single str into list of lines str
    
    pts_number = content[4].split(' ')
    pts_number = int(pts_number[0]) #number of points
    pts = content[5:pts_number+5]
    coo = np.array([[float(x) for x in p.split(' ')] for p in pts])[:,1:]
    
    elements_number = int(content[pts_number+7])
    elements = content[pts_number+8:elements_number+pts_number+8]
    elements = [[int(x) for x in e.split(' ')] for e in elements]
    tri = np.array([e[5:] for e in elements if e[1]==2])-1
    
    return coo,tri
    
if __name__ == "__main__":
    import plot_mesh
    import os
    import sys
    import find_planar_surf

    os.chdir(sys.path[0]) #change dir to main's path  

    name = '0_simple_bracket2_1.hh.msh'
    path = 'test_case/mesh/%s'%name
    p,t = read_gmsh(path)
    print(p.shape,t.shape)
    np.savez_compressed('temp/coo', coord_array=p,tri_array=t)

    #plot_mesh.plot_mesh(p, t)
    
    #label = np.load('clusters/test_case_cluster/gmsh/%s_label.npz'%name)['label']
    #label = np.load('clusters/test_case_cluster/label30.npz')['label']
    #plan_surf = find_planar_surf.find_planar_surf_manually(p, t,label)
    #print(np.amax(label))
    