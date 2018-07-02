import numpy as np

#clear = open('test_classes.txt','w')
#clear.close()

cls = np.array([[i]*1000 for i in range(9)]).flatten()


np.savetxt('nine_classes.txt',cls,fmt = '%d')