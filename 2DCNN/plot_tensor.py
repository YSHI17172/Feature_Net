import matplotlib.pyplot as plt
import numpy as np


data = np.load('/Users/User/Downloads/graph_2D_CNN/datasets/tensors/hole/node2vec_hist/hole_10:1_p=2_q=0.5_1.npy')

for i in range(5):
    fig = plt.figure(i)
    x,y = np.nonzero(data[i,:,:])
    plt.scatter(x,y)
    
    
plt.show()   
