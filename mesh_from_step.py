from read_step import read_STEP
import geometry_builder as gb
import datetime
import numpy as np
import re

def mesh(model_path,mesh_length=1,model_only=False,
        sub_type=None):

    start_time = datetime.datetime.now()

    hole_facets = [] # which face has hole
    min_length = 0 # lower length won't be seperated
    
    fname = re.findall(r'/(.+_\d+).step',model_path)[0]
    pts_array,facets = read_STEP(model_path)    
            
    #generate model
    model = gb.solid_model(pts_array,facets,hole_facets,min_length)
    
    if model_only == False:
        #generate mesh
        coord_array,tri_array = model.generate_mesh(mesh_length=mesh_length)
    else:
        coord_array = tri_array = []
        
    end_time = datetime.datetime.now()
    elapsed = (end_time-start_time).seconds
    print ("%s Mesh Created, has %d points, taken time %d seconds.\n"
    %(fname,coord_array.shape[0],elapsed))
    
    return model,coord_array,tri_array,


if __name__ == "__main__": 
    import os
    import sys

    os.chdir(sys.path[0])
    path = '3D/' #test sample path
    fname = 'slot&pocket_0.step'
    
    save_path = 'input_mesh/slot&pocket/'
    save_name = fname[:-5]
    model,coord_array,tri_array = mesh(path+fname)
    np.savez_compressed(save_path+save_name, model = model,
    coord_array=coord_array,tri_array=tri_array.astype(np.int))

    from mayavi import mlab
    mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
    #mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2],)
    mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
        tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
    mlab.show()
