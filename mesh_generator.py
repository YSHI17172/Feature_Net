# -*- coding: utf-8 -*-
import numpy as np
import geometry_builder_iso as gb
import datetime

def mesh(feature_name,mesh_length=1,random_stock=False,model_only=False,
        stock_length = 100,stock_width=100, stock_height=100,
        sub_type=None):
    np.random.seed() #reset random seed
    if random_stock == True:
        # create a block with random size range from 0.1-0.9
        w1 = np.random.randint(100,1001)*stock_width/100 # 宽
        l1 = np.random.randint(100,1001)*stock_length/100 # 长
        h1 = np.random.randint(100,1001)*stock_height/100 # 高
    else:        
        w1 = stock_width
        l1 = stock_length
        h1 = stock_height
    
    start_time = datetime.datetime.now()
    cpt_number = 0
    hole_facets = [];min_length=0
    if feature_name == 'slot':
        pts_array,facets = slot(w1,l1,h1)
    elif feature_name == 'pocket':
        pts_array,facets,hole_facets = pocket(w1,l1,h1)   
    elif feature_name == 'boss':
        min_length = 1.5*mesh_length 
        pts_array,facets,hole_facets = boss(w1,l1,h1,mesh_length)
    elif feature_name == 'step':
        pts_array,facets = step(w1,l1,h1)
    elif feature_name == 'pyramid':
        pts_array,facets,hole_facets = pyramid(w1,l1,h1)
    elif feature_name == 'protrusion':
        pts_array,facets,hole_facets = protrusion(w1,l1,h1) 
    elif feature_name == 'through_hole':
        min_length = 1.5*mesh_length 
        pts_array,facets,hole_facets = through_hole(w1,l1,h1,mesh_length)
    elif feature_name == 'blind_hole':
        min_length = 1.5*mesh_length 
        pts_array,facets,hole_facets = blind_hole(w1,l1,h1,mesh_length)
    elif feature_name == 'cone':
        min_length = 1.5*mesh_length 
        pts_array,facets,hole_facets = cone(w1,l1,h1,mesh_length)
    elif feature_name == 'dome':
        min_length = 1.5*mesh_length 
        pts_array,facets,hole_facets,cpt_number = dome(w1,l1,h1,mesh_length)
            
    #generate mesh
    model = gb.solid_model(pts_array,facets,hole_facets,min_length)
    if model_only == False:
        coord_array,tri_array = model.generate_mesh(mesh_length=mesh_length)
    else:
        coord_array = tri_array = np.array([])
    end_time = datetime.datetime.now()
    elapsed = (end_time-start_time).seconds
    print ("%s Mesh Created, has %d points, taken time %d seconds.\n"
    %(feature_name,coord_array.shape[0],elapsed))
    
    return model,coord_array,tri_array

def dome(w1,l1,h1,mesh_length):
    # 确定hole的参数
    r2 = np.random.randint(10*w1,45*w1)/100 # 半径
    cx = np.random.randint(5*w1+r2*100,95*w1-r2*100)/100 # 中心x坐标    
    cy = np.random.randint(5*l1+r2*100,95*l1-r2*100)/100 # 中心y坐标
    h1 *= 0.1
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
   
    cpt_number = int(2*np.pi*r2/mesh_length)+1 #圆模拟点数  
    
    pts_number = len(pts)
    pts.extend((r2 * np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1) # 增加圆洞口点
            for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
            
    height = [r2 * np.sin(angle)+h1 # 纬线高度
        for angle in np.linspace(0, np.pi/2, int(cpt_number/4), endpoint=False)] # 1/4 circle  

    segn_list=[]
    for h in range(1,int(cpt_number/4)):#纬线
        rad = np.sqrt(r2**2-(height[h]-h1)**2) #半径
        seg_len = np.pi*2*rad/cpt_number
        if seg_len > mesh_length:
            seg_nb = cpt_number
        else:
            seg_nb =  int(np.pi*rad*2/mesh_length)+1
        #print (seg_nb,np.pi*2*rad/seg_nb) #segmentation number of circle at different layers
        segn_list.append(seg_nb)
        pts.extend((rad * np.cos(angle)+cx, rad * np.sin(angle)+cy,height[h]) # 增加圆洞口点
                    for angle in np.linspace(0, np.pi*2, seg_nb, endpoint=False))
    
    print ('圆孔含点%d个，半圆分%d层'%(cpt_number,len(segn_list)))
    
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
 
    # 定义各面
    f_bot = [[0,3,2,1]]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+list(range(pts_number,cpt_number+pts_number))]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    
    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f

    #把圆面拆分成三角平面
    prev_divn = cpt_number
    eptsn = pts_number #counter of points
    for ia,divn in enumerate(segn_list): #纬线之间
        if divn == prev_divn: #前后分段一致
            for j in range(divn-1):#四边形,分为2个三角
                facets.extend([[eptsn+j,eptsn+prev_divn+j,eptsn+prev_divn+j+1]]) 
                facets.extend([[eptsn+j,eptsn+prev_divn+j+1,eptsn+j+1]]) 
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn+divn-1, #头尾相连
                            eptsn+prev_divn]])
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn,eptsn]])
        elif divn*2 == prev_divn: #正好2倍,分三个三角形
            for j in range(divn-1):
                facets.extend([[eptsn+2*j,eptsn+prev_divn+j,eptsn+2*j+1]]) 
                facets.extend([[eptsn+2*j+1,eptsn+prev_divn+j,eptsn+prev_divn+j+1]]) 
                facets.extend([[eptsn+2*j+1,eptsn+prev_divn+j+1,eptsn+2*j+2]]) 
            facets.extend([[eptsn+prev_divn-2,eptsn+prev_divn+divn-1,eptsn+prev_divn-1]])#头尾 
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn+divn-1,eptsn+prev_divn]])
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn,eptsn]])
        else: #开始缩小
            rest = prev_divn - divn #一组补一点，分rest 组
            sub_set = int(divn/rest) # 每组多少点
            tail = divn%rest #配对剩余点
            if tail >0:
                insert = int(rest/tail) #每几组补一点
            else:
                insert = -1
            insert_counter = 0;insert_add = 0; tail_copy = tail  
            for sub in range(rest): #分几组，每组 n 对 n+1
                insert_counter += 1
                for k in range(sub_set):
                    facets.extend([[eptsn+sub*(sub_set+1)+k+insert_add,
                                    eptsn+prev_divn+sub*sub_set+k+insert_add,
                                    eptsn+sub*(sub_set+1)+k+1+insert_add]]) 
                    facets.extend([[eptsn+sub*(sub_set+1)+k+1+insert_add,
                                    eptsn+prev_divn+sub*sub_set+k+insert_add,
                                    eptsn+prev_divn+sub*sub_set+k+1+insert_add]]) 
                #因为n+1对n, 所以要补一个三角
                facets.extend([[eptsn+(sub+1)*(sub_set+1)-1+insert_add,
                                eptsn+prev_divn+(sub+1)*sub_set+insert_add,
                                eptsn+(sub+1)*(sub_set+1)+insert_add]]) 
                #每insert组补一个正方形
                if insert_counter == insert and  tail_copy > 1:
                    facets.extend([[eptsn+(sub+1)*(sub_set+1)+insert_add,
                                    eptsn+prev_divn+(sub+1)*sub_set+insert_add,
                                    eptsn+prev_divn+(sub+1)*sub_set+1+insert_add]]) 
                    facets.extend([[eptsn+(sub+1)*(sub_set+1)+insert_add,
                                    eptsn+prev_divn+(sub+1)*sub_set+1+insert_add,
                                    eptsn+(sub+1)*(sub_set+1)+1+insert_add]])
                    insert_counter = 0; tail_copy -= 1;insert_add+=1
                                                              
            # for sub in range(rest): #分几组，每组n 对 n+1
            #     for k in range(sub_set):
            #         facets.extend([[eptsn+sub*(sub_set+1)+k,
            #                         eptsn+prev_divn+sub*sub_set+k,
            #                         eptsn+sub*(sub_set+1)+k+1]]) 
            #         facets.extend([[eptsn+sub*(sub_set+1)+k+1,
            #                         eptsn+prev_divn+sub*sub_set+k,
            #                         eptsn+prev_divn+sub*sub_set+k+1]]) 
            #     #补一个三角
            #     facets.extend([[eptsn+(sub+1)*(sub_set+1)-1,
            #                     eptsn+prev_divn+(sub+1)*sub_set,
            #                     eptsn+(sub+1)*(sub_set+1)]]) 

            if tail == 0: # 删除最后两个三角
                del facets[-1]
                del facets[-1]
            #头尾
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn+divn-1,eptsn+prev_divn]]) 
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn,eptsn]])           
 
        #print(prev_divn,eptsn)
        eptsn+=prev_divn;prev_divn=divn
         
    #补全最后三角形
    
    for i in range(seg_nb-1):
        facets.extend([[eptsn+i,pts_number-1,eptsn+i+1]])
    facets.extend([[eptsn,eptsn+i+1,pts_number-1]])
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])

    return pts_array,facets,hole_facets,cpt_number

def cone(w1,l1,h1,mesh_length):
    # 确定hole的参数
    r2 = np.random.randint(10*w1,45*w1)/100 # 半径
    d2 = np.random.randint(10*h1,90*h1)/100 # 高度
    cx = np.random.randint(5*w1+r2*100,95*w1-r2*100)/100 # 中心x坐标    
    cy = np.random.randint(5*l1+r2*100,95*l1-r2*100)/100 # 中心y坐标
    h1*=0.1
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
    (cx,cy,d2+h1)#顶点8
    ]    
   
    cpt_number = int(2*np.pi*r2/mesh_length)+1 #圆模拟点数 
    pts_number = len(pts)
    
    pts.extend((r2 * np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1) # 增加圆洞口点
            for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))

    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
    
    # 定义各面
    f_bot = [[0,3,2,1]]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+list(range(pts_number,cpt_number+pts_number))]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    
    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f

    for i in range(cpt_number-1): #把圆柱面拆分成平面
        facets.extend([[pts_number+i,pts_number-1,pts_number+1+i]])
    facets.extend([[pts_number-1,pts_number,cpt_number+pts_number-1]])
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])
    return pts_array,facets,hole_facets

def blind_hole(w1,l1,h1,mesh_length):
    # 确定hole的参数
    r2 = np.random.randint(10*w1,45*w1)/100 # 半径
    d2 = np.random.randint(10*h1,90*h1)/100 # 深度
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
    ]    
   
    cpt_number = int(2*np.pi*r2/mesh_length)+1 #圆模拟点数 
    
    pts.extend((r2 * np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1) # 增加圆洞口点
            for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
            
    pts.extend((r2* np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1-d2) # 增加底面圆点
        for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
       
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
   
    # 定义各面
    f_bot = [[0,3,2,1]]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+list(range(8,cpt_number+8))]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    #f_c1 = [list(range(8,cpt_number+8))] # 圆洞
    f_hb = [list(range(cpt_number+8,cpt_number*2+8))] #底面圆 6
    
    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f+f_hb
    
    for i in range(cpt_number-1): #把圆柱面拆分成平面
        facets.extend([[8+i,8+i+cpt_number,9+i+cpt_number,9+i]])
    facets.extend([[7+cpt_number,7+cpt_number*2,8+cpt_number,8]])
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])
    return pts_array,facets,hole_facets

def through_hole(w1,l1,h1,mesh_length):
    # 确定hole的参数
    r2 = np.random.randint(10*w1,45*w1)/100 # 半径
    #d2 = np.random.randint(10*h1,90*h1)/100 # 深度
    cx = np.random.randint(5*w1+r2*100,95*w1-r2*100)/100 # 中心x坐标    
    cy = np.random.randint(5*l1+r2*100,95*l1-r2*100)/100 # 中心y坐标
    hole_center = [[cx,cy,0],[cx,cy,h1]] # 定义洞所在面的圆心
    
    # 定义各点
    pts = [(0,0,0), #0
    (0,l1,0), #1
    (w1,l1,0),#2
    (w1,0,0),#3
    (w1,0,h1),#4
    (w1,l1,h1),#5
    (0,l1,h1),#6
    (0,0,h1),#7
    ]    
   
    cpt_number = int(2*np.pi*r2/mesh_length)+1 #圆模拟点数   
    pts.extend((r2 * np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1) # 增加圆洞口点
            for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
            
    pts.extend((r2* np.cos(angle)+cx, r2 * np.sin(angle)+cy,0) # 增加底面圆洞口点
        for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
       
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
   
    # 定义各面
    f_bot = [[0,3,2,1,-1]+list(range(cpt_number+8,cpt_number*2+8))]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+list(range(8,cpt_number+8))]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    #f_c1 = [list(range(8,cpt_number+8))] # 圆洞
    #f_hb = [list(range(cpt_number+8,cpt_number*2+8))] #底面圆 6

    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f
    
    for i in range(cpt_number-1): #把圆柱面拆分成平面
        facets.extend([[8+i,8+i+cpt_number,9+i+cpt_number,9+i]])
    facets.extend([[7+cpt_number,7+cpt_number*2,8+cpt_number,8]])
    
    #定义洞
    hole_face = [0,3]; hole_facets = []
    for i in range(len(facets)):
        if i in hole_face:
            hole_facets.append(hole_center.pop(0))
        else:
            hole_facets.append([])
    return pts_array,facets,hole_facets

def protrusion(w1,l1,h1):    
    w2 = np.random.randint(10*w1,90*w1)/100 # 宽
    l2 = np.random.randint(10*l1,90*l1)/100 # 长
    d2 = np.random.randint(10*h1,90*h1)/100 # 深度
    ls = np.random.randint(5*l1,95*l1-l2*100)/100 # 左边界
    fs = np.random.randint(5*w1,95*w1-w2*100)/100 # 前边界 
    
    h1 *=0.1
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
    (fs+w2,ls,h1+d2),#12
    (fs+w2,ls+l2,h1+d2),#13
    (fs,ls+l2,h1+d2),#14
    (fs,ls,h1+d2),#15
    ]    
       
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
   
    # 定义各面
    f_bot = [[0,3,2,1]]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+[8,9,10,11]]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    f_pbot = [[12,15,14,13]] # pocket bottom face 6
    f_pls = [[8,15,12,11]]#pocket left side 7
    f_prs = [[9,10,13,14]]#pocket right side 8
    f_pf = [[10,11,12,13]]#pocket front face 9
    f_pb = [[8,9,14,15]]#pocket back face 10

    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f+f_pbot+f_pls+f_prs+f_pf+f_pb
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])
    return pts_array,facets,hole_facets

def pyramid(w1,l1,h1):    
    # 确定参数    
    w2 = l2 = np.random.randint(10*w1,90*w1)/100 # 宽
    d2 = np.random.randint(10*h1,90*h1)/100 # 高度
    fs = ls = np.random.randint(5*l1,95*l1-l2*100)/100 # 左边界
   #fs = np.random.randint(5*w1,95*w1-w2*100)/100 # 前边界 
    
    h1*= 0.1
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
    (fs+w2/2,ls+l2/2,d2+h1), #apex 12
    (fs,ls+l2/2,h1), # 13
    (fs+w2/2,ls,h1), #14
    (fs+w2/2,ls+l2,h1),#15
    (fs+w2,ls+l2/2,h1),#16
    ]    
       
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
   
    # 定义各面
    f_bot = [[0,3,2,1]]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+[8,13,9,15,10,16,11,14]]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    f_pbot = [[8,13,9,12]] # 6
    f_pls = [[9,15,10,12]]#7
    f_prs = [[10,16,11,12]]#8
    f_pf = [[8,12,11,14]]#9

    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f+f_pbot+f_pls+f_prs+f_pf
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])
    return pts_array,facets,hole_facets


def step(w1,l1,h1):
    # 确定step的参数
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
    f_bot = [[0,3,2,1]]#底面
    f_ls = [[0,5,4,3]]#左侧面
    f_rs = [[1,2,11,10]]#右侧面
    f_lt = [[4,5,6,7]]#左顶面
    f_rt = [[8,9,10,11]]#右顶面
    f_b = [[2,3,4,7,8,11]]#front face
    f_f = [[0,1,10,9,6,5]]#back face
    f_sl = [[6,7,8,9]]#step left
    facets = f_bot+f_ls+f_rs+f_lt+f_rt+f_f+f_b+f_sl
    return pts_array,facets
        
def boss(w1,l1,h1,mesh_length):
    # 确定hole的参数
    r2 = np.random.randint(10*w1,45*w1)/100 # 半径
    d2 = np.random.randint(10*h1,90*h1)/100 # 高度
    cx = np.random.randint(5*w1+r2*100,95*w1-r2*100)/100 # 中心x坐标    
    cy = np.random.randint(5*l1+r2*100,95*l1-r2*100)/100 # 中心y坐标
    h1*= 0.1
    
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
    ]    
   
    cpt_number = int(2*np.pi*r2/mesh_length)+1 #圆模拟点数   

    pts.extend((r2 * np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1) # 增加圆洞口点
            for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
            
    pts.extend((r2* np.cos(angle)+cx, r2 * np.sin(angle)+cy,h1+d2) # 增加底面圆点
        for angle in np.linspace(0, 2*np.pi, cpt_number, endpoint=False))
       
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
   
    # 定义各面
    f_bot = [[0,3,2,1]]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+list(range(8,cpt_number+8))]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    #f_c1 = [list(range(8,cpt_number+8))] # 圆洞
    f_hb = [list(range(cpt_number+8,cpt_number*2+8))] #底面圆 6
    
    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f+f_hb
    
    for i in range(cpt_number-1): #把圆柱面拆分成平面
        facets.extend([[8+i,8+i+cpt_number,9+i+cpt_number,9+i]])
    facets.extend([[7+cpt_number,7+cpt_number*2,8+cpt_number,8]])
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])
            
    return pts_array,facets,hole_facets    
    

def pocket(w1,l1,h1):
    # 确定pocket的参数    
    w2 = np.random.randint(10*w1,90*w1)/100 # 宽
    l2 = np.random.randint(10*l1,90*l1)/100 # 长
    d2 = np.random.randint(10*h1,90*h1)/100 # 深度
    ls = np.random.randint(5*l1,95*l1-l2*100)/100 # 左边界
    fs = np.random.randint(5*w1,95*w1-w2*100)/100 # 前边界 
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
    (fs+w2,ls,h1-d2),#12
    (fs+w2,ls+l2,h1-d2),#13
    (fs,ls+l2,h1-d2),#14
    (fs,ls,h1-d2),#15
    ]    
       
    pts_array =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
   
    # 定义各面
    f_bot = [[0,3,2,1]]#底面 0
    f_ls = [[0,7,4,3]]#左侧面 1
    f_rs = [[1,2,5,6]]#右侧面 2
    f_t = [[4,7,6,5,-1]+[8,9,10,11]]#顶面 3
    f_b = [[2,3,4,5]]#front face 4
    f_f = [[0,1,6,7]]#back face  5
    f_pbot = [[12,15,14,13]] # pocket bottom face 6
    f_pls = [[8,15,12,11]]#pocket left side 7
    f_prs = [[9,10,13,14]]#pocket right side 8
    f_pf = [[10,11,12,13]]#pocket front face 9
    f_pb = [[8,9,14,15]]#pocket back face 10

    facets = f_bot+f_ls+f_rs+f_t+f_b+f_f+f_pbot+f_pls+f_prs+f_pf+f_pb
    
    #定义洞
    hole_face = 3; hole_facets = []
    for i in range(len(facets)):
        if i == hole_face:
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])
    return pts_array,facets,hole_facets
            

def slot(w1,l1,h1):
    # 确定slot的参数
    w2 = np.random.randint(10*w1,90*w1)/100 # 宽
    d2 = np.random.randint(10*h1,90*h1)/100 # 深度
    ls = np.random.randint(1*w1,95*w1-w2*100)/100 # 左边界

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
    f_bot = [[0,3,2,1]]#底面
    f_ls = [[0,1,14,15]]#左侧面
    f_rs = [[2,3,4,5]]#右侧面
    f_lt = [[12,15,14,13,-1,4,7,6,5]]#顶面
    #f_rt = [[4,7,6,5]]#右顶面
    f_b = [[0,15,12,11,8,7,4,3]]#front face
    f_f = [[1,2,5,6,9,10,13,14]]#back face
    f_sb = [[8,11,10,9]]#slot bottom
    f_sl = [[10,11,12,13]]#slot left
    f_sr = [[6,7,8,9]]#slot right
    #facets = f_bot+f_ls+f_rs+f_lt+f_rt+f_f+f_b+f_sb+f_sl+f_sr  
    facets = f_bot+f_ls+f_rs+f_lt+f_f+f_b+f_sb+f_sl+f_sr   
    return pts_array,facets

def to_parallelize(mesh_number,fname,path):
    model,coord_array,tri_array = mesh(fname)    
    np.savez_compressed(path+'/'+fname+'_%d'%(mesh_number), model = model,
    coord_array=coord_array,tri_array=tri_array.astype(np.int)) 

if __name__ == "__main__": 
    import os
    import sys
    from multiprocessing import Pool, cpu_count
    from functools import partial
    os.chdir(sys.path[0]) #change dir to main's path  
    #model,coord_array,tri_array = mesh('dome',mesh_length=1)
    
    #print ('total feature %d'%len(model.features))
    #print ('total faces %d'%len(model.faces))
    #print ('feature has %d faces'%len(model.features[0].faces))
    #print (model.features[0].faces)    

    features = ['slot','step','pocket','through_hole','blind_hole',] 
    features=['cone','pyramid','protrusion','boss']
    n_jobs = cpu_count()
    print ('using', n_jobs, 'cores')
    
    for fname in features:
        path = '/Volumes/HDD120/new_isolated/' + fname
        try:           
            os.mkdir(path)
        except FileExistsError:
            pass
        to_parallelize_partial = partial(to_parallelize,fname=fname,path=path)
        pool = Pool(processes=n_jobs)
        pool.map(to_parallelize_partial, range(5000,10000))
        pool.close()
        pool.join()
        
