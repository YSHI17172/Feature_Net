# -*- coding: utf-8 -*-
import numpy as np
import geometry_builder as gb
import commontool as ct
import os
import sys

os.chdir(sys.path[0]) #change dir to main's path 
 
# create a block with random size range from 1-10
w1 = np.random.randint(100,1001)/100 # 宽
l1 = np.random.randint(100,1001)/100 # 长
h1 = np.random.randint(100,1001)/100 # 高

w1 = l1 = h1 = 100 # 固定长宽高
mesh_number = 1 
while mesh_number < 101:
    # 确定slot的参数
    w2 = np.random.randint(10*w1,90*w1)/100 # 宽
    d2 = np.random.randint(10*h1,90*h1)/100 # 深度
    ls = np.random.randint(5*w1,95*w1-w2*100)/100 # 左边界
    
    # 定义各点
    pts = [(0,0,0), #0
    (0,l1,0), #1
    (w1,l1,0),#2
    (w1,0,0),#3
    (w1,0,h1),#4
    (w1,l1,h1),#5
    (ls+w2,l1,h1),#6
    (ls+w2,0,h1),#7
    (ls+w2,0,h1-d2),#8
    (ls+w2,l1,h1-d2),#9
    (ls,l1,h1-d2),#10
    (ls,0,h1-d2),#11
    (ls,0,h1),#12
    (ls,l1,h1),#13
    (0,l1,h1),#14
    (0,0,h1),#15
    ]
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
    
    # 定义各面
    f_bot = [[0,1,2,3]]#底面
    f_ls = [[0,1,14,15]]#左侧面~/OneDrive - University of South Carolina/New
    f_rs = [[2,3,4,5]]#右侧面
    f_lt = [[12,13,14,15]]#左顶面
    f_rt = [[4,5,6,7]]#右顶面
    f_b = [[0,3,4,7,8,11,12,15]]#front face
    f_f = [[1,2,5,6,9,10,13,14]]#back face
    f_sb = [[8,9,10,11]]#slot bottom
    f_sl = [[10,11,12,13]]#slot left
    f_sr = [[6,7,8,9]]#slot right
    facets = f_bot+f_ls+f_rs+f_lt+f_rt+f_f+f_b+f_sb+f_sl+f_sr
    
    #generate mesh
    model = gb.solid_model(pts_array,facets)
    coord_array,tri_array = model.generate_mesh()
    
    print ("Slot Mesh %d, depth %.2f, width %.2f, has %d points."%(mesh_number,d2,w2,coord_array.shape[0]))
    if coord_array.shape[0] >5000:
        print("点数过多，重来！")
        continue
    np.savez_compressed('input_mesh/slot/slot_%05d-1'%mesh_number, model = model,coord_array=coord_array,tri_array=tri_array.astype(np.int32))

    # for i in range(9): #随机旋转模型
    #     axis = ct.random_axis() # 随机旋转轴
    #     theta = np.random.uniform(0,np.pi*2) #随机旋转角度
    #     rand_axis = ct.rotation_matrix(axis,theta) #旋转矩阵
    #     new_coord_array = np.dot(rand_axis, coord_array.T).T 
    #     np.savez_compressed('input_mesh/slot/slot_%05d-%d'%(mesh_number,i+2), model = model,coord_array=new_coord_array,tri_array=tri_array.astype(np.int32))
    # 
    mesh_number +=1

# ## plot to check 
# from mayavi import mlab
# mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
# #mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2],)
# mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
#     tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
# mlab.show()




