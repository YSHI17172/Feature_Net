import numpy as np
import plot_mesh
features1 = ['slot/','step/','pocket/','blind/','through/']
features2 = ['slot','step','pocket','blind_hole','through_hole']

patch_number = 5000

i = 3

idxs =[16278, 19443, 18853, 15860, 17618, 19908, 16764, 19757, 16694, 15779, 15115]


idxs = [x-i*patch_number for x in idxs]
for fn in idxs:

    model_path = '/Volumes/ExFAT256/new_isolated/%s/%s_%d'%(features1[i],features2[i],fn)
    
    data = np.load(model_path+'.npz')
    tri_array = data['tri_array']; coord_array = data['coord_array']
    data.close()
    
    plot_mesh.plot_mesh(fn,coord_array, tri_array)

#import mayavi
#mayavi.mlab.close(all=True)