import os

d = '/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/predict/'

features = ['slot','step','boss','pocket','pyramid',
            'protrusion','through_hole','blind_hole','cone',]

files= os.listdir(d)

try:
    files.remove('.DS_Store') # read all file names in the folder
except:
    pass

names = [[] for i in range(9)]

for i,ft in enumerate(names):
    fn = features[i]
    for fil in files[:]:
        if fn in fil:
            ft.append(fil)
for i in range(9):
    for j in range(500):
        old = names[i][j]
        new = str(j+i*500)+'.txt'       
        os.rename(d+old, d+new)