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

w1 = l1 = 100 # 固定长宽高
h1 = 10

mesh_number = 1
while mesh_number < 101:
    # 确定参数    
    w2 = l2 = np.random.randint(10*w1,90*w1)/100 # 宽
    d2 = np.random.randint(10*100,90*100)/100 # 高度
    fs = ls = np.random.randint(5*l1,95*l1-l2*100)/100 # 左边界
   #fs = np.random.randint(5*w1,95*w1-w2*100)/100 # 前边界 

    hole_center = [fs+w2/2,ls+l2/2,h1]
    
    # 定义各点
    pts = [(0,0,0), #0
    (0,l1,0), #1
    (w1,l1,0),#2
    (w1,0,0),#3
    (w1,0,h1),#4
    (w1,l1,h1),#5
    (0,l1,h1),#6
    (0,0,h1),#7
    (fs,ls,h1), #8
    (fs,ls+l2,h1),#9
    (fs+w2,ls+l2,h1),#10
    (fs+w2,ls,h1),#11
    (fs+w2/2,ls+l2/2,d2), #apex 12
    (fs,ls+l2/2,h1), # 13
    (fs+w2/2,ls,h1), #14
    (fs+w2/2,ls+l2,h1),#15
    (fs+w2,ls+l2/2,h1),#16
    ]    
       
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
   
    # 定义各面
    f_bot = [[0,1,2,3]]#底面 0
    f_ls = [[0,3,4,7]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,5,6,7,-1]+[8,13,9,15,10,16,11,14]]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    f_pbot = [[8,13,9,12]] # 6
    f_pls = [[9,15,10,12]]#7
    f_prs = [[10,16,11,12]]#8
    f_pf = [[8,14,11,12]]#9

    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f+f_pbot+f_pls+f_prs+f_pf
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])

    #generate mesh
    coord_array,tri_array = mesh_generator.mesh_tri(pts_array,facets,hole_facets)
    print ("Pyramid Mesh %d, height %.2f, length %.2f, has %d points."%(mesh_number,d2,w2,coord_array.shape[0]))
    
    if coord_array.shape[0] >3000:
        print("点数过多，重来！")
        continue
    
    np.savez_compressed('input_mesh/pyramid/pyramid_%05d-1'%mesh_number, coord_array=coord_array,tri_array=tri_array.astype(np.int32))
    
    for i in range(9): #随机旋转模型
        axis = ct.random_axis() # 随机旋转轴
        theta = np.random.uniform(0,np.pi*2) #随机旋转角度
        rand_axis = ct.rotation_matrix(axis,theta) #旋转矩阵
        new_coord_array = np.dot(rand_axis, coord_array.T).T 
        np.savez_compressed('input_mesh/pyramid/pyramid_%05d-%d'%(mesh_number,i+2), 
        coord_array=new_coord_array,tri_array=tri_array.astype(np.int32))
    
    mesh_number += 1

# # # # plot to check 
# from mayavi import mlab
# mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
# # mlab.plot3d(pts_array[:, 0], pts_array[:, 1], pts_array[:, 2],tube_radius=0.2)
# mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
#     tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
# mlab.show()




