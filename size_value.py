# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cluster
import read_dat
import matplotlib.cm as mcm

part_name = "ISO/Protrusion.dat"
coord_array, tri_array = read_dat.read_dat(part_name)

clusters = np.load('temp/clusters.npz')['clusters']

#get adjcency matrix 
cluster_adjmap = cluster.get_cluster_adj_matrix(clusters[-2,:],tri_array) 

fcl = [1,2,3,8,9,10,11,20,22,27,30,37,43,50,56] # cluster in feature
nvl = [] # number of vertices in the cluster
cvl = [] # cluster value list 
doml = []

cm = np.linspace(0,1,len(fcl)) #get a set of different floats for different colors
np.random.shuffle(cm) #shuffle numbers, avoid similar colors too close to each other

cmap = mcm.get_cmap('Set3') # get instance of colormap, for read single color

for fc in fcl:
    nv = np.count_nonzero(clusters[-2,:]==fc) #统计cluster内有多少点
    nvl.append(nv)    
    p = np.where(clusters[-2,:]==fc)[0][0] # cluster内第一个点号
    cv = clusters[-1,p] # cluster value
    cvl.append(cv)
    adjcl = np.where(cluster_adjmap[fc-1,:] ==1)[0]+1 # 相邻cluster list
    dom = adjcl.size # 统计相邻总数= degree of freedom
    doml.append(dom)

for i in list(range(len(fcl))):    
    adjcl = np.where(cluster_adjmap[fcl[i]-1,:] ==1)[0]+1 # 相邻cluster list
    plt.text(doml[i]*.99,cvl[i],'%i'%fcl[i],color='red') # 图上添加编号
    for adjc in adjcl:
        if adjc in fcl and adjc > fcl[i]:  #plot edges
            plt.plot([doml[i],doml[fcl.index(adjc)]],[cvl[i],cvl[fcl.index(adjc)]], c = cmap(cm[i]))

# plot nodes
plt.scatter(doml,cvl,s = np.array(nvl)*50,c=cm,cmap='Set3',marker='o',lw=1,edgecolor='black')

plt.show()