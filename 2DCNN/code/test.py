from subprocess import call
import numpy as np
import os
import sys
import tempfile
import shutil
import igraph
from sklearn.decomposition import PCA

os.chdir(sys.path[0]) #change dir to main's path  

path_node2vec = '/Users/User/Downloads/snap/examples/node2vec/'

adj_mat = np.loadtxt('/Users/User/Downloads/graph_2D_CNN/datasets/data_as_adj/test1/slot_1.txt')
g = igraph.Graph.Adjacency(adj_mat.tolist(),mode='UNDIRECTED')
my_edgelist = igraph.Graph.get_edgelist(g)
 # create temp dir to write and read from
tmpdir = tempfile.mkdtemp()
# create subdirs for node2vec
os.makedirs(tmpdir + '/graph/')
os.makedirs(tmpdir + '/emb/')
# write edge list
with open(tmpdir + '/graph/input.edgelist', 'w') as my_file:
    my_file.write('\n'.join('%s %s' % x for x in my_edgelist))
# execute node2vec
p = '2'; q='0.5'

call([path_node2vec + 'node2vec -i:' + tmpdir + 
'/graph/input.edgelist' + ' -o:' + tmpdir + '/emb/output.emb' +
' -p:' + p + ' -q:' + q],shell=True)

# read back results
emb = np.loadtxt(tmpdir + '/emb/output.emb',skiprows=1)
# sort by increasing node index and keep only coordinates
emb = emb[emb[:,0].argsort(),1:]

shutil.rmtree(tmpdir)

my_pca = PCA(n_components=20)
pca_output = my_pca.fit_transform(emb)