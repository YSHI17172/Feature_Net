import os
import sys

os.chdir(sys.path[0])

p = 'results/parameter/'
r = 'results/'
ic = 'incorrect/'
mod = 'models/'

parameters= os.listdir(p)
results = os.listdir(r)
incorrects = os.listdir(ic)
models = os.listdir(mod)

try:
    parameters.remove('.DS_Store') 
    results.remove('.DS_Store') 
    incorrects.remove('.DS_Store')
    models.remove('.DS_Store')
except:
    pass

for f in parameters[:]:
    if ':' in f:
        new = f.replace(':','-')
        os.rename(p+f, p+new)
        
for f in incorrects:
    if 'npz' in f:
        if ':' in f:
            new = f.replace(':','-')
            os.rename(ic+f, ic+new)
    else:
        sub_path = ic + f + '/'
        for subf in  os.listdir(sub_path):
            if ':' in subf:
                new = subf.replace(':','-')
                os.rename(sub_path+subf, sub_path+new)

for f in models:
    if 'h5' in f:
        if ':' in f:
            new = f.replace(':','-')
            os.rename(ic+f, ic+new)
    else:
        sub_path = mod + f + '/'
        for subf in  os.listdir(sub_path):
            if ':' in subf:
                new = subf.replace(':','-')
                os.rename(sub_path+subf, sub_path+new)
    
for f in results[:]:
    if '.' in f:
        if ':' in f:
            new = f.replace(':','-')
            os.rename(r+f, r+new)
    else:
        sub_path = r + f + '/'
        for subf in  os.listdir(sub_path):
            if ':' in subf:
                new = subf.replace(':','-')
                os.rename(sub_path+subf, sub_path+new)
    
