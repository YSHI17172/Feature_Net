import numpy as np
from read_from_gmsh import read_gmsh
import plot_planar_surf
import half_edge
from read_dat import read_dat

fname = 'DEMO04_1.hh_0.5.dat'

path = 'test_case/mesh/%s'%fname
label_path = 'test_case/label/%s_label.npz'%fname
#label_path = 'clusters/test_case_cluster/00001/%s_label.npz'%fname

if 'dat' in fname:
    coord_array, tri_array = read_dat(path)
else:
    coord_array, tri_array = read_gmsh(path)
label = np.load(label_path)['label']

# Build numpy structured array half edge for an efficient computation
M = half_edge.half_edge(coord_array, tri_array)

faces = [9,58]

nl = np.amax(label).astype(int)   
plnsrf = np.zeros(shape=(np.amax(tri_array)+1, 1), dtype=int)

for i in range(1, nl+1):         
    vtx_in_label = np.unique(M['facet'][np.nonzero(label == i)[0], :]) # points index
    if i in faces:
        plnsrf[vtx_in_label] = i
    else:
        plnsrf[vtx_in_label] = 0

plot_planar_surf.plot_planar_surf(coord_array, tri_array, plnsrf)


data = {'0_simple_bracket2_1.hh.msh':{'STEP':[[3,14],[9,10]],'SLOT':[[5,6,7]],'THROUGH':[[1],[2]]},
'1_spacer_66_1.hh.msh':{'STEP':[[18,23],[19,24]],'THROUGH':[[25],[14],[13],[12],[11]],'BLIND':[[5,7],[1,8],[4,6],[3,10],[2,9]]},
'goodpart1_1.hh_0.5.dat':{'STEP':[[11,12,35],[18,19,36],[25,26,37],[3,29,34]],'SLOT':[[14,15,16],[21,22,23],[2,7,28],[8,9,31]],'THROUGH':[[32],[33],[38],[39],[40]]},
'sipe_socket_1.hh_1.dat':{'STEP':[[2,4],[3,5],[7,8],[6,17]],'SLOT':[[11,12,13,19,20,21],[22,24,30,18,28,29,23,25,14,26,27]],'THROUGH':[[31],[33],[34],[35],[36],[37],[38]]},
'DEMO04_1.hh_0.5.dat':{'STEP':[[15,20]],'POCKET':[[42,46,47,48,84,87],[43,44,45,49,85,86],[23,24,25,26,68,69,70,71,50],[19,28,29,30,31,32,33,34,35,36,37,38,39,72,73,74,75,76,77,78,79,80,81,82,83]],
'THROUGH':[[63],[64],[88],[89]],'BLIND':[[13,61],[8,57],[14,62],[7,56],[40,66],[1,54],[2,55],[41,67],[12,60],[27,65],[9,58],[5,10],[11,59],[3,4]]}}
