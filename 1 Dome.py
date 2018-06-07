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

for mesh_number in range(1,2):
    # 确定hole的参数
    r2 = np.random.randint(10*w1,45*w1)/100 # 半径
    cx = np.random.randint(5*w1+r2*100,95*w1-r2*100)/100 # 中心x坐标    
    cy = np.random.randint(5*l1+r2*100,95*l1-r2*100)/100 # 中心y坐标
    hole_center = [cx,cy,h1] # 定义洞所在面的圆心
    
    # 定义各点
    pts = [(0,0,0), #0
    (0,l1,0), #1
    (w1,l1,0),#2
    (w1,0,0),#3
    (w1,0,h1),#4
    (w1,l1,h1),#5
    (0,l1,h1),#6
    (0,0,h1),#7
    (cx,cy,h1+r2)#8dome apex
    ]    
   
    cpt_number = 64 #圆模拟点数 
    factm,trin = ct.divby2(cpt_number) #用2分解  
    pts_number = len(pts)
    min_length = int(np.pi*2*r2/cpt_number)+1 #set min_length for mesh    
    pts.extend((r2 * np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1) # 增加圆洞口点
            for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
            
    height = [r2 * np.sin(angle)+h1 # 纬线高度
        for angle in np.linspace(0, np.pi/2, int(cpt_number/4), endpoint=False)]    

    for h in range(1,int(cpt_number/4)-factm):#纬线
        rad = np.sqrt(r2**2-(height[h]-h1)**2)
        pts.extend((rad * np.cos(angle)+cx, rad * np.sin(angle)+cy,height[h]) # 增加圆洞口点
           for angle in np.linspace(0, np.pi*2, cpt_number, endpoint=False))
    for i in range(factm):#收缩分段 32-16-8      
        rad = np.sqrt(r2**2-(height[h+i+1]-h1)**2)
        pts.extend((rad * np.cos(angle)+cx, rad * np.sin(angle)+cy,height[h+i+1]) # 增加圆洞口点
           for angle in np.linspace(0, np.pi*2, int(cpt_number/(2**(i+1))), endpoint=False))   

    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
 
    # 定义各面
    f_bot = [[0,1,2,3]]#底面 0
    f_ls = [[0,3,4,7]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,5,6,7,-1]+list(range(pts_number,cpt_number+pts_number))]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    
    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f
    
    #把圆面拆分成平面
    for i in range(int(cpt_number/4)-1-factm): #纬线之间
        for j in range(cpt_number-1):#经纬四边形
            f_c_seg = [[pts_number+i*cpt_number+j,pts_number+i*cpt_number+j+1,
                        pts_number+j+1+(i+1)*cpt_number,pts_number+j+(i+1)*cpt_number]]
            facets.extend(f_c_seg)
        facets.extend([[pts_number+(i+1)*cpt_number-1,pts_number+i*cpt_number, #头尾
                    pts_number+(i+1)*cpt_number,pts_number+(i+2)*cpt_number-1]])
                
    en = int(pts_number+((cpt_number/4)-factm-1)*cpt_number) #完整圆点数
    for i in range(factm):#收缩分段 64-32-16-8
        for j in range(int(cpt_number/(2**(i+1)))-1):#32-16-8
            f_c_seg = [[en+j*2,en+j*2+1,en+j*2+2, 
                        en+int(cpt_number/(2**i))+j+1,
                         en+int(cpt_number/(2**i))+j]]
            facets.extend(f_c_seg)
        facets.extend([[en+int(cpt_number/(2**i))-2,en+int(cpt_number/(2**i))-1,en,#头尾
                        en+int(cpt_number/(2**i)),
                        en+int(cpt_number/(2**i))+int(cpt_number/(2**(i+1)))-1]])
        en += int(cpt_number/(2**i))
         
    #补全最后三角形
    for i in range(trin-1):
        f_c_seg = [[en+i,en+i+1,pts_number-1]]
        facets.extend(f_c_seg)
    facets.extend([[en,en+i+1,pts_number-1]])
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])

    #generate mesh
    coord_array,tri_array = mesh_generator.mesh_tri(pts_array,facets,hole_facets,min_length)
    print ("Dome Mesh%d, radius: %f, has %d points"%(mesh_number,r2,coord_array.shape[0]))
#     np.savez_compressed('input_mesh/dome/dome_%05d-1'%mesh_number, 
#     coord_array=coord_array,tri_array=tri_array.astype(np.int32))
# 
#     for i in range(9): #随机旋转模型
#         axis = ct.random_axis() # 随机旋转轴
#         theta = np.random.uniform(0,np.pi*2) #随机旋转角度
#         rand_axis = ct.rotation_matrix(axis,theta) #旋转矩阵
#         new_coord_array = np.dot(rand_axis, coord_array.T).T 
#         np.savez_compressed('input_mesh/dome/dome_%05d-%d'%(mesh_number,i+2),
#         coord_array=new_coord_array,tri_array=tri_array.astype(np.int32))
# 
# # # plot to check 
# from mayavi import mlab
# mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
# #mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2],)
# mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
#     tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
# mlab.show()
# 

