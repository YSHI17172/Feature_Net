# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy.sparse import coo_matrix,csc_matrix,csgraph
from scipy.spatial import cKDTree
import scipy as sp
from itertools import combinations

def divby2(n):
    """
    divide a number by 2 repeatly,until it no large than 8
    return how many times it was divided and the rest number
    """
    m=0
    while n > 8 :
        if n%2 == 0:
            m+=1
            n = n/2
        else:
            break
    return m,int(n)

def find_normal(pts):
    ptscoord = np.copy(pts)
    center_g = np.mean(ptscoord, axis=0).flatten() # 重心/坐标平均值
    ptscoord -= center_g # 标准化
    u, s, vh = np.linalg.svd(ptscoord, compute_uv=True)
    v = vh.conj().T  # Take transpose to get same result as Matlab svd
    v = np.column_stack((v[:, 2], v[:, 1], v[:, 0])) # 倒序，0 为主vector
    v[np.abs(v)<1e-9] = 0  # 好看一点，省的好多小数
    return v
    
def find_normal2(pts):
    """
    去除重心计算
    """
    ptscoord = np.copy(pts)
    u, s, vh = np.linalg.svd(ptscoord, compute_uv=True)
    v = vh.conj().T  # Take transpose to get same result as Matlab svd
    v = np.column_stack((v[:, 2], v[:, 1], v[:, 0])) # 倒序，0 为主vector
    v[np.abs(v)<1e-9] = 0  # 好看一点，省的好多小数
    return v

def project_to_plane(point,plane): # 多点投影到2D平面
    x = np.dot(point,plane[:,1])
    y = np.dot(point,plane[:,2])
    return np.column_stack((x,y))  

def to3d(pts,v,c): # 2D 坐标转换为 3D
    # v: 平面向量, c 平面离O点距离 
    pn = pts.shape[0] ;new = np.zeros((pn,3)) # 预建    
    z = c*v[:,0] # 0 为法向量
    for p in range(pn): # loop over all points
       x = v[:,1] * pts[p,0]
       y = v[:,2] * pts[p,1]
       new[p,:] = x+y+z      
    return new
    
def to3d_v2(pts,v,c): # 2D 坐标转换为 3D, 矩阵计算版本，提高速度
    # v: 平面向量, c 平面离O点距离 
    pn = pts.shape[0]
    temp = np.column_stack((pts,np.ones(pn)*c)) #add normal vector value   
    v2 = np.column_stack((v[:, 1], v[:, 2], v[:, 0])).conj().T  # 修正顺序，保证xyz
    new = np.dot(temp,v2) # 2d 投影 3d 空间
    return new


def connect_pts(pts,breakpoint): #把平面边界点序，转化为符合trimesh的格式
    edges = []; pn =len(pts)
    for i in range(pn-1): # [0,1,2,3,4]
        temp = (i,i+1) #[(0,1),(1,2),(2,3),(3,4,)]
        edges.append(temp)
    end = (pn-1,0) # 补全头尾相连 [4,0]
    edges.append(end)  
    if breakpoint > 0: #loop 2 exsits
        edges[breakpoint-1] = (breakpoint-1,0)
        edges[-1] = (pn-1,breakpoint)
    return edges

def round_trip_connect(pts): # 不重新编序,线两端按点号大小排列
    edges = []; pn =len(pts) 
    start = pts[0] #line loop start point
    for i in range(pn-1): # [0,1,2,3,4,-1,5,6,7,8]
        if pts[i] < 0: #跳过分割标记 -1
            pass
        else:
            if pts[i+1]<0: #使用-1分割同一面中不同的line loop
                if pts[i] < start:
                    temp = (pts[i],start)
                elif pts[i] > start:
                    temp = (start,pts[i])
                else:
                    raise ValueError('The definition of facet is wrong, start and end are the same, please check face %s.'%pts)                
                start = pts[i+2] #start point of next line loop
            else:        
                if pts[i]< pts[i+1]:
                    temp = (pts[i],pts[i+1]) #[(0,1),(1,2),(2,3),(3,4,)]
                elif pts[i] > pts[i+1]:
                    temp = (pts[i+1],pts[i])
                else:
                    raise ValueError('The definition of facet is wrong, start and end are the same, please check face %s.'%pts)                
            edges.append(temp)
            
    if pts[pn-1] < start :   
        end = (pts[pn-1],start) # 补全头尾相连 [0，4]
    elif pts[pn-1] > start:
        end = (start,pts[pn-1])
    else:
        raise ValueError('The definition of facet is wrong, start and end are the same, please check face %s.'%pts)
    
    edges.append(end)  
    return edges

def round_trip_connect_original(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]

def find_line_angle(facets,two_face,pts_list):    
    pts1 = facets[two_face[0]] #face 1 point list
    l1 = pts_list[pts1[1]]-pts_list[pts1[0]];l2 = pts_list[pts1[2]]-pts_list[pts1[1]]
    norm1 = np.cross(l1,l2)
    pts2 = facets[two_face[1]] #face 1 point list
    l3 = pts_list[pts2[1]]-pts_list[pts2[0]];l4 = pts_list[pts2[2]]-pts_list[pts2[1]]
    norm2 = np.cross(l3,l4)
    angle = np.dot(norm1,norm2)/(np.linalg.norm(norm1) * np.linalg.norm(norm2))   
    #find two points not on the edges
    p1 = pts_list[np.setdiff1d(pts1, pts2)[0]]
    p2 = pts_list[np.setdiff1d(pts2, pts1)[0]]
    convex = np.dot((p2-p1),norm1)
    degree = np.arccos(angle)*180/np.pi
    if convex > 0: #conve edge
        degree+=180
    return angle,degree

def line_segements_reorder(fi,facets,facets_lines_copy,lines_list_seg,lines_array):
    """
    fucntion to put segmented lines back into the face
    and reconnect them in correct order
    """
    face = facets_lines_copy[fi] #get line index in the face 
    face_seg = [] #转face中线段号为连续点号，以便输入.但需要检查线段方向  
    break_point_loop2 = -1 #used for record loop2
    
    #检查第一条线有没有反 
    head1= lines_list_seg[face[0]][0] #线段1首位点
    end1 = lines_list_seg[face[0]][-1] #线段1末尾点
    if end1 in lines_list_seg[face[1]]: # 没反
        face_seg.extend(lines_list_seg[face[0]][:-1]) # save 线段1
        end = lines_list_seg[face[0]][-1] # 记录尾点以便后续核对
    elif head1 in lines_list_seg[face[1]]: #反了
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
            if -1 in facets[fi] and i < len(face)-2: #此面有2段线,-2 because it needs at least three lines to form a loop
                break_point_loop2 = len(face_seg)
                if lines_list_seg[face[i]][0] in lines_list_seg[face[i+1]]:#check first line of second loop
                    face_seg.extend(lines_list_seg[face[i]][-1:0:-1]) #倒序
                    end = lines_list_seg[face[i]][0]
                elif lines_list_seg[face[i]][-1] in lines_list_seg[face[i+1]]: 
                    face_seg.extend(lines_list_seg[face[i]][:-1])
                    end = lines_list_seg[face[i]][-1]
            else:
                raise ValueError('连接面%d的线段%d和线段%d时候，头尾点%s和%s对不上，请检查。当前末点%d'
                    %(fi,face[i-1],face[i],lines_array[face[i-1]],lines_array[face[i]],end))
    return face_seg,break_point_loop2

def LineSegments_3d(P1,P2,num_points=10,edge_length=-1):
  
  number_points=num_points
  if edge_length>0:
    p1=np.array(P1)
    p2=np.array(P2)
    number_points=np.floor(np.sqrt(np.sum((p2-p1)**2))/edge_length)+1
    if number_points < 2:
        number_points = 2
  
  t=np.linspace(0,1,number_points)
  points=[(P1[0]+param*(P2[0]-P1[0]),P1[1]+param*(P2[1]-P1[1]),P1[2]+param*(P2[2]-P1[2])) for param in t]
  vertices=[(j,j+1) for j in range(0,len(points)-1,1)]
  return points,vertices;

def line_segmentation_linear(li,ref_face,P1,P2,pts_segments,lines_array,coord_array,facets_lines):
    """
    为了合理减少mesh点，在分割边界时，参照对面边的距离分割。
    """
    ref_lines = facets_lines[ref_face][:] #参照面内边,[:]means copy this list
    ref_lines.remove(li)#排除所在边
    for ref_line in ref_lines[:]:#排除邻边,[:]means copy this list,不然remove以后由于长度减少，就会跳过一个index
        p1_line2 = coord_array[lines_array[ref_line,0],:];p2_line2 = coord_array[lines_array[ref_line,1],:] #目标边坐标            
        dist = distance_segment_to_segment(P1,P2,p1_line2,p2_line2) #find distance
        if dist < 1e-9:
            ref_lines.remove(ref_line)   
                                              
    new_pts = [pts_segments[0]]; # for save result
    new_pts_segments = pts_segments[:]#copy for start                         
    progress = 0; line_length = np.linalg.norm(P1 - P2) # loop parameter
    p2_seg = new_pts_segments[1]
    while progress < line_length:          
        p1_seg = new_pts[-1]; #新起点         
        seg_length = np.linalg.norm(np.array(p1_seg) - np.array(p2_seg)) #线段长度

        min_dist_all = []    #找到最小距离边       
        for ref_line in ref_lines:                
            p1_line2 = coord_array[lines_array[ref_line,0],:];p2_line2 = coord_array[lines_array[ref_line,1],:] #目标边坐标            
            dist = distance_segment_to_segment(p1_seg,p2_seg,p1_line2,p2_line2) #find distance
            min_dist_all.append(dist)
        min_dist = np.amin(min_dist_all)#最小距离
        
        if (min_dist > 10*seg_length) or (min_dist < 5*seg_length): # 细了或者粗了
            new_pts_segments,new_line_segments = LineSegments_3d(p1_seg,P2,edge_length = min_dist/10.) # 重新分                       
            new_pts.append(new_pts_segments[1])
        else:
            new_pts.append(new_pts_segments[1]) 
            new_pts_segments = new_pts_segments[1:] #不重新分           
        
        if len(new_pts_segments) > 1: #in case the last point    
            p2_seg = new_pts_segments[1] #新末点
        progress = np.linalg.norm(np.array(P1) - np.array(new_pts[-1]))

    #print (len(pts_segments),len(new_pts),line_length)
    line_segments = [(j,j+1) for j in range(0,len(new_pts)-1,1)]
    return new_pts,line_segments

def line_segmentation_3d(pts_list,lines_array,facets_lines,line_max_rate,line_max_rate_face,facet_min_length,min_length):
    """
    分割边长，以便生成mesh
    改进版本 3d 分割
    """
    coord_array = np.copy(pts_list) #copy 
    lines_list_seg = [[]for i in range(lines_array.shape[0])] # for save new lines

    # 分割边线
    for li in range(lines_array.shape[0]):  
        p1 = lines_array[li,0];p2 = lines_array[li,1] # point index
        P1 = coord_array[p1,:];P2 = coord_array[p2,:] # coordinates of points
        seg_points = line_max_rate[li] #分段数量
        pts_segments,line_segments = LineSegments_3d(P1,P2,num_points=seg_points)  #segmentation 
        
        ref_face = line_max_rate_face[li] #参照面--含有短边
        if seg_points > 24 and facet_min_length[ref_face] > min_length and len(facets_lines[ref_face]) > 3: # 分段过多时，进行调整.排除含有圆边面,排除三角形         
            pts_segments,line_segments=line_segmentation_linear(li,ref_face,P1,P2,pts_segments,lines_array,coord_array,facets_lines)
        
        offset = coord_array.shape[0]-1 # find 点序号 修正准备 
        line_segments_array =  np.array(([list(p) for p in line_segments])) #为方便加减数字转化为array
        line_segments_array += offset #修正序号       
        new_line = list(np.unique(line_segments_array)) #flattened and sorted, start from 0 [0,1,2...]
        
        #复原首尾点序号
        #line_segments_array[0,0] = edge[0]; line_segments_array[-1,-1] = edge[1] 
        new_line[0] = p1;new_line[-1] = p2 
        lines_list_seg[li] = new_line #save new segments as line

        #merge new points
        pts_segments_array = np.array(([list(p) for p in pts_segments])) # 为转3d转array
        coord_array = np.vstack((coord_array,pts_segments_array[1:-1,:])) # merge new seg point

    return coord_array,lines_list_seg

def is_equal(a,b):
    if abs(a-b) < 1e-9:
        return True
    return False

def distance_segment_to_segment(p1_line1,p2_line1,p1_line2,p2_line2):
    """
    input format [x,y,z]
     解析几何通用解法，可以求出点的位置，判断点是否在线段上
     算法描述：设两条无限长度直线s、t,起点为s0、t0，方向向量为u、v
     最短直线两点：在s1上为s0+sc*u，在t上的为t0+tc*v
     记向量w为(s0+sc*u)-(t0+tc*v),记向量w0=s0-t0
     记a=u*u，b=u*v，c=v*v，d=u*w0，e=v*w0——(a)；
     由于u*w=、v*w=0，将w=-tc*v+w0+sc*u带入前两式得：
     (u*u)*sc - (u*v)*tc = -u*w0  (公式2)
     (v*u)*sc - (v*v)*tc = -v*w0  (公式3)
     再将前式(a)带入可得sc=(be-cd)/(ac-b2)、tc=(ae-bd)/(ac-b2)——（b）
     注意到ac-b2=|u|2|v|2-(|u||v|cosq)2=(|u||v|sinq)2不小于0
     所以可以根据公式（b）判断sc、tc符号和sc、tc与1的关系即可分辨最近点是否在线段内
     当ac-b2=0时，(公式2)(公式3)独立，表示两条直线平行。可令sc=0单独解出tc
     最终距离d（L1、L2）=|（P0-Q0)+[(be-cd)*u-(ae-bd)v]/(ac-b2)|
    """
    x1=p1_line1[0];y1=p1_line1[1];z1=p1_line1[2]
    x2=p2_line1[0];y2=p2_line1[1];z2=p2_line1[2]
    x3=p1_line2[0];y3=p1_line2[1];z3=p1_line2[2]
    x4=p2_line2[0];y4=p2_line2[1];z4=p2_line2[2]
    
    ux = x2 - x1;uy = y2 - y1;uz = z2 - z1;
    vx = x4 - x3;vy = y4 - y3;vz = z4 - z3;
    wx = x1 - x3;wy = y1 - y3;wz = z1 - z3;

    a = (ux * ux + uy * uy + uz * uz); #u*u
    b = (ux * vx + uy * vy + uz * vz); #u*v
    c = (vx * vx + vy * vy + vz * vz); #v*v
    d = (ux * wx + uy * wy + uz * wz); #u*w 
    e = (vx * wx + vy * wy + vz * wz); #v*w
    dt = a * c - b * b;
    
    sd = dt;td = dt;

    sn = 0.0;#sn = be-cd
    tn = 0.0;#tn = ae-bd
    
    if is_equal(dt,0): #两直线平行
        sn = 0  #在s上指定取s0
        sd = 1.00;   #防止计算时除0错误
        tn = e;      #按(公式3)求tc
        td = c
    else:
        sn = (b * e - c * d);
        tn = (a * e - b * d);
        if sn <0: #最近点在s起点以外，同平行条件
            sn = 0
            tn = e
            td = c
        elif sn > sd: #最近点在s终点以外(即sc>1,则取sc=1)
            sn = sd
            tn = e+b
            td = c
    if tn < 0: #最近点在t起点以外
        tn = 0
        if (-d < 0): #按(公式2)计算，如果等号右边小于0，则sc也小于零，取sc=0
            sn = 0 
        elif -d > a: #按(公式2)计算，如果sc大于1，取sc=1
            sn = sd
        else:
            sn = -d
            sd = a
    elif tn > td:
        tn = td
        if ((-d+b)<0):
            sn = 0
        elif ((-d+b)>a):
            sn = sd
        else:
            sn = (-d+b)
            sd = a
            
    sc = 0; tc = 0
    
    if is_equal(sn,0):
        sc = 0
    else:
        sc = sn/sd
    if is_equal(tn,0):
        tc = 0
    else:
        tc = tn/td
    
    dx = wx + (sc * ux) - (tc * vx);
    dy = wy + (sc * uy) - (tc * vy);
    dz = wz + (sc * uz) - (tc * vz);
    return math.sqrt(dx * dx + dy * dy + dz * dz)

"""
meshpy.triangle.build(mesh_info, 
verbose=False, Verbose: Gives detailed information about what Triangle is doing.vertex-by-vertex 
refinement_func=None, Imposes a user-defined constraint on triangle size
attributes=False, Assigns an additional attribute to each triangle that identifies
                    what segment-bounded region each triangle belongs to.
volume_constraints=True, Imposes a maximum triangle area.
max_volume=None, Imposes a specific number of maximum triangle area
allow_boundary_steiner=True, Prohibits the insertion of Steiner points on the mesh boundary
allow_volume_steiner=True,  prohibits the insertion of Steiner points on any segment, 
quality_meshing=True, Quality mesh generation with no angles smaller than 20 degrees.
generate_edges=None, same as generate_faces
generate_faces=False, Outputs (to an .edge file) a list of edges of the triangulation
min_angle=None, impose a min_angle 
mesh_order=None, 
generate_neighbor_lists=False) Outputs (to a .neigh file) a list of triangles neighboring each triangle.
"""


def DoTriMesh(points,vertices,edge_length=-1,holes=[],tri_refine=None):
    import meshpy.triangle as triangle
    info = triangle.MeshInfo()
    info.set_points(points)
    if len(holes)>0:
        info.set_holes(holes)
    info.set_facets(vertices)
    
    if tri_refine!=None:
        mesh = triangle.build(info,refinement_func=tri_refine,
        allow_boundary_steiner=False)
    elif edge_length<=0:
        mesh = triangle.build(info,allow_boundary_steiner=False)   
    else:
        mesh = triangle.build(info,max_volume=0.5*edge_length**2,
        allow_boundary_steiner=False)
    
    mesh_points = np.array(mesh.points)
    mesh_elements = np.array(mesh.elements)
    # import matplotlib.pyplot as plt  
    # plt.figure()
    # plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_elements,) 
    # plt.show()
    return mesh_points,mesh_elements 

def needs_refinement(vertices, area ):
    vert_origin, vert_destination, vert_apex = vertices
    bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
    bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3

    dist_center = math.sqrt( bary_x**2 + bary_y**2 )
    max_area = 100*(math.fabs( 0.002 * (dist_center-0.5) ) + 0.0001)
    return area > max_area


def connection_matrix(tri_array):
    i = np.vstack((tri_array[:,0],tri_array[:,1],tri_array[:,2])).ravel()
    j = np.vstack((tri_array[:,1],tri_array[:,2],tri_array[:,0])).ravel()
    I = np.concatenate((i,j))
    J = np.concatenate((j,i))

    nopts = np.amax(tri_array)+1
    notri = tri_array.shape[0]

    adjpts = coo_matrix((np.ravel(np.ones(shape = (6*notri, 1))),\
    (I.ravel(), J.ravel())),shape = (nopts,nopts),dtype=np.int8)
    
    adjpts += np.eye(nopts)

    return csc_matrix((adjpts>0)*1)

def find_scd(con_mat_in):
    #find the shortest links between to points in a adjacency matrix
    B = con_mat_in
    C = B
    E = C
    D = np.ones(con_mat_in.shape)
    F = np.ones(con_mat_in.shape)
    j = 0  # Index shift to start at zero
    while not (E.nnz == E.shape[0]**2):
        if j == con_mat_in.shape[0]:
            break
        E = C
        D = D+F-(C >= 1).astype(float)  # D = D+(C == 0).astype(float) works, but is slow because of == 0
        C = C.dot(B)
        j += 1

    return D

"""
Toolbox for generating a mesh
"""

# Extract the edges
# ouput, edges and boundary edges 
def FindEdges(t):
  #pdb.set_trace();  
  NE=t.shape[0]
  # generate an array of all edges
  tt=np.array([t[:,0],t[:,1],t[:,1],t[:,2],t[:,2],t[:,0]]).T.reshape(3*NE,2)
  ttt=np.sort(tt,1)
  
  # find all boundary edges
  all_edges=[ tuple(x) for x in ttt ]
  boundary_edges=[x for x in all_edges if all_edges.count(x)==1]
  
  # find all unique edges
  all_edges=list(set(all_edges))
  return all_edges,boundary_edges;


##################
#
#  Boundary Tools
#
##################

# given one segment 
# e.g.  (X,2) find segment (2,Y) and delete (2,Y) from list 
def FindNextSegment(all_segments,node):
  # find next connecting segment  
  help=[x for x in all_segments if x[0]==node]   
  
  new_bound=False
  if len(help)==0: #if connecting segment does not exist (=>new boundary) 
    ret=all_segments[0]
    new_bound=True    
  else:
    ret=help[0]
  
  del all_segments[all_segments.index(ret)]
  return ret,new_bound;  
  
  
# sort segments:  (3,6),(6,1),(1,12),(12,5),...
# on output: sorted segments and indices of the different boundaries
def SortSegments(all_segments):  
  count=len(all_segments)

  node=-1
  sorted_segments=[]
  boundaries=[]
  for j in range(len(all_segments)):
    seg,new_bound=FindNextSegment(all_segments,node)
    node=seg[1]
    sorted_segments.append(seg)    
    if new_bound==True:
      boundaries.append(j)
    
  if len(sorted_segments)!=count:
    print("Something is wrong, number of segments not the same")   
  return sorted_segments,boundaries;

# connect segments in a defined way
# (see SortSegments), but start sorting with a defined point p
# multiple p'2 for different closed boundaries are possible
def ConnectBoundary(boundary_segments,Pall,p=[]):
  
  # sort the boundary segments  
  allseg=boundary_segments[:]  
  allseg,boundaries=SortSegments(allseg)
  if p==[]:
    return allseg,boundaries;
    
  max_boundaries=len(boundaries)
   
  # find all nodes on the given boundary
  nodes=[x[0] for x in allseg]
  # find closest nodes to desired point list p  
  indices,distances=FindClosestNode(nodes,Pall,p)
  
  #change order within each closed boundary
  flag_sorted=[]
  for j in range(len(boundaries)):
   flag_sorted.append(False) 
   
  for j in range(len(indices)):
    # find position of node in the boundary list
    # indj gives the position of the segment in allseg
    indj=nodes.index(indices[j])
    # find the number of boundary the node belongs to
    this_boundary=(np.where((np.array(boundaries)<=indj))[0])[-1]
    
    if flag_sorted[this_boundary]==False:
      # define the indices for slicing      
      ind_1=boundaries[this_boundary]
      if this_boundary+1==max_boundaries:
        ind_2=len(allseg)
      else:
        ind_2=boundaries[this_boundary+1]  
      
      # rearange the segments in the corresponding boundary     
      allseg=allseg[:ind_1]+allseg[indj:ind_2]+allseg[ind_1:indj]+allseg[ind_2:]
      # resort only once      
      flag_sorted[this_boundary]=True
  
  return allseg,boundaries;


#
# find closest node to point p0 in a list of N nodes
# Pall coordinates of M nodes  M>=N
# constraint defines constraints on distance
def FindClosestNode(nodes,Pall,p0,constraint=-1,tree=None):
  # take those points of the node list
  
  if tree==None:
    p_nodes=np.array(Pall)
    p_nodes=p_nodes[nodes] 
    # look for minimum distance, define dist function
    mytree = cKDTree(p_nodes)
  else:
    mytree=tree
    
  dist, index = mytree.query(np.array(p0))
  
  node_closest=[nodes[j] for j in index]
   
  # check constraints
  num_p= len(p0)
  if constraint<0:
    return node_closest,dist;
  elif np.isscalar(constraint)==True:
    constraint=constraint*np.ones(num_p)
  elif len(p0)!=len(constraint):
    print('Error in constraint definition')
    return [],[]
  
  # check constraint for each node
  flags=[((dist[j]<=constraint[j]) | (constraint[j]<0)) for j in range(num_p)]
  for j in range(num_p):
    if flags[j]==False:
      node_closest[j]=-1
  return node_closest,dist;
  
   
# check relative position of two points   
def SamePoint(p1,p2,delta):
  dp=(np.array(p1)-np.array(p2))
  d=np.sqrt(dp[0]**2+dp[1]**2)
  ret=False  
  if d<delta:
    ret=True
  return ret;

#####################
#
# Make simple curves
#
#####################
#
#
# 
# make a circle or part of it  
#
def CircleSegments(middle,radius,num_points=10,a_min=0.,a_max=2.*np.pi,edge_length=-1):  
  # check for closed loop
  number_points=num_points
  if edge_length>0:
    number_points=np.floor(abs(radius/edge_length*(a_max-a_min)))+1
    
  delta=(a_max-a_min)/number_points  
  closed=False;  
  if abs(a_max-a_min-2*np.pi)<0.1*delta:
    closed=True
    
  t=np.linspace(a_min,a_max,number_points,not closed)
  # define points
  points=[(middle[0]+radius*np.cos(angle),middle[1]+radius*np.sin(angle)) for angle in t]
  
  # define vertices
  vertices=[(j,j+1) for j in range(0,len(points)-1,1)]    
  if closed==True:
    vertices+=[(len(points)-1,0)]
  return points,vertices;



# Straight line
def LineSegments(P1,P2,num_points=10,edge_length=-1):
  
  number_points=num_points
  if edge_length>0:
    p1=np.array(P1)
    p2=np.array(P2)
    number_points=np.floor(np.sqrt(np.sum((p2-p1)**2))/edge_length)+1
  
  t=np.linspace(0,1,number_points)
  points=[(P1[0]+param*(P2[0]-P1[0]),P1[1]+param*(P2[1]-P1[1])) for param in t]
  vertices=[(j,j+1) for j in range(0,len(points)-1,1)]
  return points,vertices;


# Rectangle
def RectangleSegments(P1,P2,num_points=60,edge_length=-1):
  P11=[P2[0],P1[1]]
  P22=[P1[0],P2[1]]  
  npoints=np.floor(num_points/4)
  p_1,v_1=LineSegments(P1,P11,npoints,edge_length)
  p_2,v_2=LineSegments(P11,P2,npoints,edge_length)  
  p_3,v_3=LineSegments(P2,P22,npoints,edge_length)
  p_4,v_4=LineSegments(P22,P1,npoints,edge_length)
  p,v=AddSegments(p_1,p_2)
  p,v=AddSegments(p,p_3)
  p,v=AddSegments(p,p_4)
  return p,v


# List of points
def PointSegments(p):
  p1=np.array(p)
  delta=np.min(np.sqrt(np.sum((p1[1:]-p1[:-1])**2,axis=1)))
  Pall=[(x[0],x[1]) for x in p]  
  closed=False  
  if SamePoint(p1[0],p1[-1],delta)==True:
    Pall=Pall[:-1]  
    closed=True    
    
  vertices=[(j,j+1) for j in range(0,len(Pall)-1,1)]
  if closed==True:
    vertices+=[(len(Pall)-1,0)]  
  
  return Pall,vertices

#Connect two different polygons  
def AddSegments(P1,P2,closed=False):  
  p1=np.array(P1)
  p2=np.array(P2)
  # find smallest distance within points p1 and p2
  min1=np.min(np.sqrt(np.sum((p1[1:]-p1[:-1])**2,axis=1)))
  min2=np.min(np.sqrt(np.sum((p2[1:]-p2[:-1])**2,axis=1)))
  delta=np.min([min1,min2])
  
  # Add second curve to first curve 
  del_first=SamePoint(p1[-1],p2[0],delta)
  Pall=P1[:]  
  if del_first==True:
    Pall+=P2[1:]
  else:
    Pall+=P2
  
  # check if Pall is closed 
  del_last=SamePoint(Pall[-1],p1[0],delta)
  if del_last==True:
    Pall=Pall[:-1]
    
  vertices=[(j,j+1) for j in range(0,len(Pall)-1,1)]
  if (del_last==True) or (closed==True):
    vertices+=[(len(Pall)-1,0)]
  
  return Pall,vertices;  


# Append Curves
def AddCurves(p1,v1,p2,v2):
  # make one list  
  p=p1+p2
  v2n=[(v2[j][0]+len(p1),v2[j][1]+len(p1)) for j in range(len(v2))]
  v=v1+v2n
  return p,v;


#deprecated,unfinished
# def update_edge_points(face,mesh_lines):
#     #每mesh一个面后，检查新创点，如果落在边界上，更新每个面的边界
#         if mesh_pts.shape[0] > 0:
#             cmat = ct.connection_matrix(mesh_lines) # csc sparse matrix 
#             #cmat_index = cmat.nonzero() # n*2 index matrix 
#             for i in range(1,len(face)+1):
#                 p1 = face[i-1]; p2 = face[i]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def random_axis():
    """
    Return random unit axis
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return (x,y,z)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a = 10;b = 15; c =25; d = 200
    x = np.random.randint(50,size = 20)-25
    y = np.random.randint(50,size = 20)-25
    z = (d-a*x-b*y)/c
    coo = np.column_stack((x,y,z))

    v = find_normal(coo)
    ptscoo2 = project_to_plane(coo,v)
    plane_dist = np.mean(np.dot(coo,v[:,0])) 
    
#     
#     pts = [(0,0,0), #0
#     (0,15,0), #1
#     (100,100,0),#2
#     (15,0,0),#3
#     ]    
#        
#     pts_array =  np.array(([list(p) for p in pts])).astype(np.float)
#     v = find_normal(pts_array)
#     v2 = find_normal2(pts_array)
# 
#     print(v)
#     print(v2)
#     print(v==v2)

    # 验证to3d功能
    #recover = to3d(ptscoo2,v,plane_dist)
    #recover_v2 = to3d_v2(ptscoo2,v,plane_dist)
    #print (coo)
    #print (ptscoo2)
    #print (recover==recover_v2)
    
    #验证conn_mat
    # tri = np.random.randint(20,size=9).reshape((3,3))
    # tri = np.array([[0,1,4],[4,2,3],[0,3,4]])
    # cmat = connection_matrix(tri)
    # dist,path = csgraph.shortest_path(cmat,directed=False,return_predecessors=True)
    # 
    # #print (cmat.nonzero())
    # #print (find_scd(csc_matrix(cmat)))
    # #print (csgraph.floyd_warshall(cmat))
    # print (dist)
    # print (path)
    #
    
    #p,v = LineSegments([0,0],[10,10],num_points=10,edge_length=-1)

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # p,v = LineSegments_3d([1,2,3],[5,4,1],num_points=10,)
    # pts = np.array(([list(x) for x in p])) 
    # ax.plot(pts[:,0],pts[:,1],pts[:,2])
    # plt.show()

    # theta = math.pi /2 
    # axisz = [0,0,1]
    # axisx = [1,0,0]    
    # v = [0,10,0]    
    # print (np.dot(rotation_matrix(axisx,theta), v))
    # 
    # print (np.dot(rotation_matrix(axisx,theta), coo)) 