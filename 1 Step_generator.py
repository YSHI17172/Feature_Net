# -*- coding: utf-8 -*-
import numpy as np
import mesh_generator
import commontool as ct
import os
import sys

os.chdir(sys.path[0]) #change dir to main's path 

# create a block with random size range from 1-10
w1 = np.random.randint(100,1001)/100 # 宽
l1 = np.random.randint(100,1001)/100 # 长
h1 = np.random.randint(100,1001)/100 # 高

w1 = l1 = h1 = 100 # 固定长宽高

for mesh_number in range(1,101):
    # 确定slot的参数
    w2 = np.random.randint(10*w1,90*w1)/100 # 宽
    d2 = np.random.randint(10*h1,90*h1)/100 # 深度
    
    # 定义各点
    pts = [(0,0,0), #0
    (0,l1,0), #1
    (w1,l1,0),#2
    (w1,0,0),#3
    (w1,0,h1),#4
    (0,0,h1),#5
    (0,l1-w2,h1),#6
    (w1,l1-w2,h1),#7
    (w1,l1-w2,h1-d2),#8
    (0,l1-w2,h1-d2),#9
    (0,l1,h1-d2),#10
    (w1,l1,h1-d2),#11
    ]
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
    
    # 定义各面
    f_bot = [[0,1,2,3]]#底面
    f_ls = [[0,3,4,5]]#左侧面
    f_rs = [[1,2,11,10]]#右侧面
    f_lt = [[4,5,6,7]]#左顶面
    f_rt = [[8,9,10,11]]#右顶面
    f_b = [[2,3,4,7,8,11]]#front face
    f_f = [[0,1,10,9,6,5]]#back face
    f_sl = [[6,7,8,9]]#step left
    facets = f_bot+f_ls+f_rs+f_lt+f_rt+f_f+f_b+f_sl
        
    #generate mesh
    coord_array,tri_array = mesh_generator.mesh_tri(pts_array,facets)
    print ("Step Mesh %d, height %.2f, length %.2f, has %d points."%(mesh_number,d2,w2,coord_array.shape[0]))

    np.savez_compressed('input_mesh/step/step_%05d-1'%mesh_number, coord_array=coord_array,tri_array=tri_array.astype(np.int32))

    for i in range(9): #随机旋转模型
        axis = ct.random_axis() # 随机旋转轴
        theta = np.random.uniform(0,np.pi*2) #随机旋转角度
        rand_axis = ct.rotation_matrix(axis,theta) #旋转矩阵
        new_coord_array = np.dot(rand_axis, coord_array.T).T 
        np.savez_compressed('input_mesh/step/step_%05d-%d'%(mesh_number,i+2), coord_array=new_coord_array,tri_array=tri_array.astype(np.int32))

# # plot to check 
# from mayavi import mlab
# mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
# ##mlab.plot3d(pts_array[:, 0], pts_array[:, 1], pts_array[:, 2],tube_radius = .5)
# mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
#     tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
# mlab.show()
# 




    
    
    