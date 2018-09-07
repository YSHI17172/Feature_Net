import numpy as np
import geometry_builder as gb
from multiprocessing import Pool, cpu_count
from functools import partial

def run(f_name,model_path):
    data = np.load(model_path+f_name+'.npz')
    tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
    step_model = data['step_model'][()]
    data.close()
    
    min_length = 1.5
    
    pts_array = step_model.coo_array 
    facets =  step_model.facets 
    hole_facets =  step_model.hole_in_facets 
    
    #generate model
    my_model = gb.solid_model(pts_array,facets,hole_facets,min_length)

    np.savez_compressed(model_path+f_name, step_model=step_model, my_model=my_model,
    coord_array=coord_array,tri_array=tri_array.astype(np.int))
        
if __name__ == "__main__": 
 
    feature2 = 'through_hole_blind_hole'
    feature1 = 'through_hole_blind_hole'
    
    model_path = 'input_mesh/intersecting/%s/'%feature1
    files = [feature2 + '_' + str(i) for i in range(59,5000)]
    
    # for fname in files:
    #     run(fname,model_path)
        
    n_jobs = cpu_count()-1
    to_parallelize_partial = partial(run,model_path=model_path)
    pool = Pool(processes=n_jobs)
    pool.map(to_parallelize_partial,files)
    pool.close()
    pool.join()
    