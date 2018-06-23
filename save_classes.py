import numpy as np

#clear = open('test_classes.txt','w')
#clear.close()

cls = np.array([[i]*100 for i in range(3)]).flatten()


np.savetxt('test_classes.txt',cls,fmt = '%d')