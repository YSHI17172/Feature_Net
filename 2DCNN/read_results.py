import os
import json
import numpy as np
import re
#path_to_results = 'C:\\Users\\yshi\\OneDrive - University of South Carolina\\New\\Feature_Net\\graph_2D_CNN\\datasets\\results\\'

dataset = 'new_isolated_c20_cluster'

path_to_results = '/Users/User/OneDrive - University of South Carolina/New/Feature_Net/2DCNN/results/%s/'%dataset
path_to_parameter = '/Users/User/OneDrive - University of South Carolina/New/Feature_Net/2DCNN/results/parameter/'

results_names = os.listdir(path_to_results)
parameter_names = os.listdir(path_to_parameter)
try:
    results_names.remove('.DS_Store')
except:
    pass

#remove folders
for f in results_names[:]:
    if '.' not in f:
        results_names.remove(f)

my_prec = 4 # desired precision

for name in results_names:
    time_stamp = re.findall(r'augmentation_(.+)_results',name)[0]
    print ('=======',name,'=======')
    for pnm in parameter_names:
        if time_stamp in pnm:
            para_name=pnm
            with open(path_to_parameter + para_name, 'r') as my_file:
                parameter = json.load(my_file)
                        
                for key in parameter:
                    print(key,':',parameter[key])
                print(para_name)
    with open(path_to_results + name, 'r') as my_file:
        tmp = json.load(my_file)
    vals = [elt[1] for elt in tmp['outputs']] # 'outputs' contains loss, accuracy for each repeat of each fold 
    vals = [val*100 for val in vals]    
    print ('mean:', round(np.mean(vals),my_prec))
    print ('median:', round(np.median(vals),my_prec))
    print ('max:', round(max(vals),my_prec))
    print ('min:', round(min(vals),my_prec))
    print ('stdev', round(np.std(vals),my_prec))
    histories = tmp['histories']


    
