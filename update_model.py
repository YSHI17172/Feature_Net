import numpy as np
import geometry_builder as gb
from multiprocessing import Pool, cpu_count
from functools import partial
from read_STEP import read_STEP,closed_shell

def run(f_name,step_path,model_path,save_path):
   
    data = np.load(model_path+f_name+'.npz')
    tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
    step_model = data['step_model'][()]
    data.close()
    
    min_length = 1.5
    
    # step_data = read_STEP(step_path+f_name+'.step') 
    # step_model = closed_shell(step_data)
    # pts_array,facets,hole_facets = step_model.get_facets()
    pts_array = step_model.coo_array
    facets = step_model.facets
    hole_facets = step_model.hole_in_facets
    
    #generate model
    my_model = gb.solid_model(pts_array,facets,hole_facets,min_length)

    np.savez_compressed(save_path+f_name, step_model=step_model, my_model=my_model,
    coord_array=coord_array,tri_array=tri_array.astype(np.int))
        
if __name__ == "__main__": 
 
    feature2 = 'through_hole_pocket'
    feature1 = 'through_hole_pocket'
    
    model_path ='/Volumes/ExFAT256/intersecting/%s/'%feature1
    step_path = 'STEP/%s/'%feature1
    save_path = '/Volumes/ExFAT256/new/%s/'%feature1
    files = [feature2 + '_' + str(i) for i in range(0,5000)]
    
    # for fname in files:
    #     run(fname,model_path)
        
    n_jobs = 7
    to_parallelize_partial = partial(run,step_path=step_path,model_path=model_path,save_path=save_path)
    pool = Pool(processes=n_jobs)
    pool.map(to_parallelize_partial,files)
    pool.close()
    pool.join()
    