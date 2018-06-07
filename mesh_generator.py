# -*- coding: utf-8 -*-
import numpy as np
import commontool as ct

def mesh_tri(pts_list,facets):
    """
    input: pts - list of points' corrdinate [(x1,y1,z1),(x2,y2,z2)...]
            facets - list of facets' edge points in order [[0,1,2,3,],[2,3,4,5]....]
    output: coord_array - coordinates of mesh points in numpy array shape=(n,3)
            tri_array - triangles of the mesh, array shape (n,3)    
    """
    
    #定义线条
    lines_list = [];facets_lines = []
    for face in facets:    
        edges = ct.round_trip_connect(face) #connect all points
        lines_list.extend(edges) # save lines
        facets_lines.append(edges)  # 把facets 转化成 线为元素
    lines_array = np.unique(lines_list,axis=0) #储存所有的线 [[0,1],[0,3],...]
    
    #把facets中以 （0，1）元素转化为 lines_array 中对应线的index 
    facets_lines_copy = []#copy it so we can use it twice
    for face in facets_lines: # [[0,2,4,6],[0,3,29,30],....]
        for i in range(len(face)):
            line = np.where((lines_array==(face[i])).all(axis=1))
            face[i] = int(line[0])
        facets_lines_copy.append(list(face))
    
    #pts_list =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
    
    #获得线条长度
    lines_length = np.ones(lines_array.shape[0])*-1 #预建
    for i in range(lines_array.shape[0]):
        line_coo = pts_list[lines_array[i]] #获得坐标
        distance = np.linalg.norm(line_coo[0,:]- line_coo[1,:]) #计算长度
        lines_length[i] = distance #储存
    facet_min_length = np.array([np.amin(lines_length[face]) for face in facets_lines]) # 计算面内最小边长
    
    #一条边属于两个面，找到在哪个面中与短边比值最大
    line_max_rate = []
    for li in range(lines_array.shape[0]):
        two_face = []
        for fi in range(len(facets_lines)):
            if li in facets_lines[fi]:
                two_face.append(fi)
        max_rate = int(np.amax(lines_length[li]/facet_min_length[two_face])*10)
        line_max_rate.append(max_rate)
        
    coord_array = np.copy(pts_list); lines_list_seg = [[]for i in range(lines_array.shape[0])] # for save new lines
    
    # 分割边线
    for fi in range(len(facets)):    
        #get 平面向量
        face = facets[fi] # [0,1,2,3]
        ptscoord_plane = pts_list[face, :] # get coordinates of vertices    
        v = ct.find_normal(ptscoord_plane) # find vectors of plane   
        plane_dist = np.mean(np.dot(ptscoord_plane,v[:,0])) # distance from the origin to the plane
        
        while facets_lines[fi]:
            edge_ind = facets_lines[fi][0] # get edge index 
            edge = lines_array[edge_ind,:] # get start and end points index   
            ptscoord = pts_list[edge] # get 3d coord         
            ptscoord_edge2d = ct.project_to_plane(ptscoord,v) # get 2d corrdinates  
            ptscoord_plane3 = list(tuple(i) for i in ptscoord_edge2d) #把numpy array 转化为tuple list
    
            P1 = ptscoord_plane3[0]; P2 = ptscoord_plane3[1] # get 2d coordinates
            
            #按长度与面内最小边长度比*10 为分割点数
            seg_points = line_max_rate[edge_ind]
            
            pts_segments,line_segments = ct.LineSegments(P1,P2,num_points=seg_points) #segmentation   
                            
            offset = coord_array.shape[0]-1 # find 点序号 修正准备 
            line_segments_array =  np.array(([list(p) for p in line_segments])) #为方便加减数字转化为array
            line_segments_array += offset #修正序号       
            new_line = list(np.unique(line_segments_array))
            
            #复原首尾点序号
            #line_segments_array[0,0] = edge[0]; line_segments_array[-1,-1] = edge[1] 
            new_line[0] = edge[0];new_line[-1] = edge[1]
            lines_list_seg[edge_ind] = new_line
    
            #merge new points
            pts_segments_array = np.array(([list(p) for p in pts_segments])) # 为转3d转array
            new_seg_pts = ct.to3d_v2(pts_segments_array[1:-1,:],v,plane_dist) # convert to 3d coo
            coord_array = np.vstack((coord_array,new_seg_pts)) # merge new seg point
            
            for  face_remove in facets_lines:
                try:
                    face_remove.remove(edge_ind) # 去掉完成分割的线
                except ValueError:
                    pass
    
    tri_array = np.zeros((0,3)) # 储存坐标和三角
    
    for fi in range(len(facets)):
        
        face = facets_lines_copy[fi] #get line index in the face 
    
        face_seg = [] #转face中线段号为连续点号，以便输入.但需要检查线段方向  
        
        #检查第一条线有没有反 
        head1= lines_list_seg[face[0]][0] #线段1首位点
        end1 = lines_list_seg[face[0]][-1] #线段1末尾点
        head2 =  lines_list_seg[face[1]][0] #线段2首位点
        end2 = lines_list_seg[face[1]][-1] #线段2末尾点
        if end1 == head2 or end1 == end2: # 没反
            face_seg.extend(lines_list_seg[face[0]][:-1]) # save 线段1
            end = lines_list_seg[face[0]][-1] # 记录尾点以便后续核对
        elif head1 == head2 or head1 == end2: #反了
            face_seg.extend(lines_list_seg[face[0]][-1:0:-1]) #倒序save 
            end = lines_list_seg[face[0]][0]# 记录尾点以便后续核对
        else: #对不上
            raise ValueError('连接面%d的线段%d和线段%d时候，头尾点%s和%s对不上，请检查。'
                            %(fi,face[0],face[1],lines_array[face[0]],lines_array[face[1]]))
        
        for i in range(1,len(face)): # 检查后续线段
            if end == lines_list_seg[face[i]][0]: # 正常
                face_seg.extend(lines_list_seg[face[i]][:-1]) 
                end = lines_list_seg[face[i]][-1] # 记录尾点以便后续核对 
            elif end == lines_list_seg[face[i]][-1]: # 反了
                face_seg.extend(lines_list_seg[face[i]][-1:0:-1])  # 倒序处理
                end = lines_list_seg[face[i]][0] # 记录尾点以便后续核对
            else: #对不上
                raise ValueError('连接面%d的线段%d和线段%d时候，头尾点%s和%s对不上，请检查。当前末点%d'
                    %(fi,face[i-1],face[i],lines_array[face[i-1]],lines_array[face[i]],end))
    
        ptscoord = coord_array[face_seg, :] # get coordinates of vertices
        
        v = ct.find_normal(ptscoord) # find vectors of plane
        
        ptscoord2 = ct.project_to_plane(ptscoord,v) # get 2d corrdinates
        
        ptscoord3 = list(tuple(i) for i in ptscoord2) #把numpy array 转化为tuple list
        
        plane_dist = np.mean(np.dot(ptscoord,v[:,0])) # distance from the origin to the plane
        
        edges = ct.connect_pts(face_seg) #输入准备  
    
        edge_leng = np.amin(lines_length[face])/5. #set mesh length
    
        mesh_pts, mesh_lines = ct.DoTriMesh(ptscoord3,edges,edge_length = edge_leng) # create mesh
    
        opn = len(face_seg); offset = coord_array.shape[0] - opn # 点序号 修正准备
            
        mesh_pts = ct.to3d_v2(mesh_pts[opn:,:],v,plane_dist) # convert to 3d coo    
    
        coord_array = np.vstack((coord_array,mesh_pts)) # merge new mesh point
    
        mesh_lines[np.where(mesh_lines>(opn-1))] += offset #序号修正
        
        mesh_lines_temp = np.copy(mesh_lines) # copy，防止loop出错
    
        for p in range(opn): # convert vertices id in new meshto original id
            pind = np.where(mesh_lines_temp==p) 
            mesh_lines[pind] = face_seg[p]
    
        tri_array = np.vstack((tri_array,mesh_lines)) #merge triangles
    return coord_array,tri_array

if __name__ == "__main__":

    # create a block with random size range from 1-10
    w1 = np.random.randint(100,1001)/100 # 宽
    l1 = np.random.randint(100,1001)/100 # 长
    h1 = np.random.randint(100,1001)/100 # 高
    
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
    
    coord_array,tri_array = mesh_tri(pts_array,facets)
    
    # check points
    from mayavi import mlab
    mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
    #mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2],)
    mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
        tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
    mlab.show()




