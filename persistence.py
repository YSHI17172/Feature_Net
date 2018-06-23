# -*- coding: utf-8 -*-
import numpy as np
import math
import datetime 

def persistence(HKS):
    """ Determine persistence value k and level s per part point
    :param HKS: array of size (#points x #steps) with heat values per point per time step
    :param per: persistence threshold value used to determine persistence value and level per point. optional param
    :return v: array of shape=(#points, 2) with persistence level k in v[:, 0] and persistence value s in v[:, 1]
    """

    print("Calculating persistence..")
    starttime_total = datetime.datetime.now()
    
    hsd = np.std(HKS,axis=0) # 标准差
    rhsd = hsd / np.mean(HKS,axis = 0) # 变异系数
    rhsdd = np.append(rhsd[1:],0)- rhsd # 微分
    p1 = np.argmax(rhsdd) # 变异系数导数最大点
    p2 = np.argmin(np.abs(rhsdd[p1:]))+p1 #变异系数高点

    #per = np.amax(HKS[:,p2]) # P2最大HKS
    per = np.mean(HKS[:,p2])  #P2平均HKS
    
    print ('Thresh is %f'%per)
    
    # Check if per value is given, else take it from HKS max
    if per == 0:
        thresh = HKS[:, (HKS.shape[1] - 1)].max()  
    else:
        thresh = per

    n = HKS.shape[0]
    v = np.zeros(shape = (n, 2))

    # Loop over points and calculate persistence level and value
      
    for index in range(n):
        # Find persistence level k
        test_thresh = (HKS[index, :] <= thresh)
        # If only True or False in test_thresh and first one is false
        if len(np.unique(test_thresh)) == 1 and not test_thresh[0]:
            # No HKS value in this row is higher than thresh, so np.nonzero returns an empty array
            k = HKS.shape[1]-1
        else:
            k = np.nonzero(HKS[index, :] <= thresh)[0][0] - 1

        # Persistence value s
        s = math.fsum(HKS[index, 0:k+1])  # Use fsum instead of sum to avoid loss of precision

        # Store persistence level k and value s in v array
        v[index, 0] = k
        v[index, 1] = s
    
    endtime_total = datetime.datetime.now()  
    elapsedtime_total = (endtime_total - starttime_total).seconds
    print ("Persistence caculation complete.Total elapsed time is %s seconds.\n"%elapsedtime_total)

    return v
