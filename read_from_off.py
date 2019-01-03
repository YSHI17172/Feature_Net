# -*- coding: utf-8 -*-
import numpy as np

def read_off(path):
    
    with open(path,'r') as infile:
        content=infile.read()#将所有stl数据读到content中, str
    content = content.splitlines() #split single str into list of lines str
    
    if 'OFF' not in content[0]: #check format
        raise TypeError('Input format is not OFF!')
    
    numbers = content[1].split(' ')
    pts_num = int(numbers[0]) #number of points
    tri_num = int(numbers[1]) #number of facets
    
    pts = content[2:pts_num+2]
    
    pts_array = np.array([np.array([float(x) for x in p.split(' ')]) for p in pts])
    
    pts_array = pts_array[:,:3]
    
    tris = content[pts_num+2:]
    
    tri_array =  np.array([np.array([int(x) for x in t.split(' ')]) for t in tris])
    
    tri_array = tri_array[:,1:]
    
    return pts_array,tri_array
    
if __name__ == "__main__":
    import plot_mesh
    import find_planar_surf
    
    name = '5_good-bracket_0.5.off'
    path = 'input_mesh/test_case/%s'%name
    p,t = read_off(path)
    print(p.shape,t.shape)
    #np.savez_compressed('temp/coo', coord_array=p,tri_array=t)
    plot_mesh.plot_mesh(p, t)
    
    label = np.load('clusters/test_case_cluster/%s_label.npz'%name)['label']
    plan_surf = find_planar_surf.find_planar_surf_manually(p, t,label)
    
