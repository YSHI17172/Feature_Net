# -*- coding: utf-8 -*-
import numpy as np
import commontool as ct

class solid_model():
    def __init__(self,pts_list,facets,hole_in_facet=[],min_length=0):
        self.pts_list = pts_list # numpy array
        self.facets = facets # list in list
        self.min_length = min_length
        self.hole = hole_in_facet
        self.lines = find_lines(facets,pts_list) #all line objects
        self.faces = find_face(pts_list,facets,self.lines) # all face objects        
        find_line_attributes(self.faces,self.lines,min_length,facets,pts_list) # update line attributes
        find_loop_concavity(self.faces,self.lines) #update loop concavity
        find_adj_face(self.faces,self.lines) # update adjacent faces info for every face
        self.features = find_features(self.faces,self.lines) # collect faces to find form feature
      
    def generate_mesh(self):
        coord_array,lines_list_seg = ct.line_segmentation_3d(self.pts_list,self.lines,self.faces,self.min_length)  
        coord_array,tri_array =tri_mesh(self.facets,self.faces,self.lines,coord_array,lines_list_seg,self.min_length,self.hole)
        return coord_array,tri_array
                                                              
class face():
    def __init__(self,fid,f,pts_list,lines):
        self.ID = fid
        self.points_id = f
        self.loops = find_loop(f,pts_list,lines)
        self.min_length = find_min_length(self.loops,lines) 
        self.adjacent_faces = None #list
        self.norm = find_face_norm(f,pts_list) 
        self.type = find_face_type(self,pts_list) # -1 inner, 1 outer
        find_loop_type(self.type,self.loops,pts_list,lines) # update loop type   
        self.mesh_points = None # numpy array to save all mesh points index in the face    
        self.mesh_triangles = None # numpy array               
                                                            
class loop():
    def __init__(self,lp,pts_list,lines):
        self.line_list = find_line_list(lp,lines)  # only list of index, no objects
        self.type = None # 1,outer,-1,inner,0,hybrid
        self.concavity = None # 1 convex, -1 concave, 0 hybrid, 2 transitional

class line():
    def __init__(self,eid,e,pts_list):
        self.ID = eid
        self.start = e[0]
        self.end = e[1]
        self.length = np.linalg.norm(pts_list[e[0],:]- pts_list[e[1],:])
        self.vector = pts_list[e[1],:]- pts_list[e[0],:] #end - start
        self.seg_rate = None
        self.max_rate_face =None
        self.angle = None
        self.parent_faces = None # list 
        self.type = None # -1 inner, 1 outer
        self.segments = None # list [0,15,16,17,18,1] new points of segmentation

class point():
    def __init__(self,p):
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]

class feature():
    def __init__(self,faces_in_feature,loop):
        self.faces = faces_in_feature
        if loop.type == -1 and loop.concavity == 1:
            self.type = 1 # convex inner feature, such as protrusion
        elif loop.type == -1 and loop.concavity == -1:
            self.type = 2 # concave inner feature, such as pocket, blind hole
        elif loop.type == 0 and loop.concavity == 0:
            self.type = 3 # outside feature       
        else:
            self.type = -1 #undefined

def find_features(faces,lines):
    features = []
    for fc in faces:
        for lp in fc.loops: 
            faces_in_feature = [fc.ID]
            if lp.type == -1: # inner loop  
                for ln in lp.line_list:
                    for fs in lines[ln].parent_faces:
                        if fs != fc.ID: 
                            faces_in_feature.append(fs)                                        
                            find_next_adj_face(fs,faces,faces_in_feature)                   
            elif lp.type == 0: #hybrid loop has inner edge
                for ln in lp.line_list:
                    if lines[ln].type == -1: # inner line
                        for fs in lines[ln].parent_faces:
                            if fs != fc.ID:
                                faces_in_feature.append(fs)                                    
                                find_next_inner_adj_face(fs,faces,faces_in_feature)
            faces_in_feature = np.setdiff1d(faces_in_feature,fc.ID) # remove base face
            if duplicate_check(faces_in_feature,features) and len(faces_in_feature) > 0:
                features.append(feature(faces_in_feature,lp)) # create object 
    return features

def duplicate_check(faces_in_feature,features):
    for f in features:
        if set(faces_in_feature) == set(f.faces):
            return False
    return True
                       
def find_next_inner_adj_face(fs,faces,faces_in_feature):
    """collect adjacent faces recursively, only add inner face"""
    for adj_fs in faces[fs].adjacent_faces:
        if (adj_fs not in faces_in_feature) and faces[adj_fs].type == -1:
            faces_in_feature.append(adj_fs)
            find_next_inner_adj_face(adj_fs,faces,faces_in_feature)  
                             
def find_next_adj_face(fs,faces,faces_in_feature):   
    """collect adjacent faces recursively"""
    for adj_fs in faces[fs].adjacent_faces:
        if adj_fs not in faces_in_feature:
            faces_in_feature.append(adj_fs)
            find_next_adj_face(adj_fs,faces,faces_in_feature)                                                      
                                                   
def find_adj_face(faces,lines):
    """find adjacent faces for every face"""
    for fa in faces:
        adj_face = []
        for lp in fa.loops:
            for ln in lp.line_list:
                adj_face.extend(lines[ln].parent_faces)
        fa.adjacent_faces = np.setdiff1d(adj_face,fa.ID)   
                                    
def find_loop_concavity(faces,lines):
    """ check the concavity of the loop"""
    for fc in faces:
        for lp in fc.loops:
            loop_angle = []
            for ln in lp.line_list:
                loop_angle.append(lines[ln].angle)
            loop_angle = np.array(loop_angle)
            if np.all(loop_angle>180): # all convex edges
                lp.concavity = 1
            elif np.all(loop_angle<180): # all concave edges
                lp.concavity = -1
            elif np.any(loop_angle == 180): # any tangent edge
                lp.concavity = 2
            else: # hybrid
                lp.concavity = 0     

def find_face_type(face,pts_list):
    """ check whether the face is stock face """
    pts_dists = []
    n = face.norm
    for pi in range(pts_list.shape[0]):
        if pi not in face.points_id:
            dist = np.dot((pts_list[pi,:]-pts_list[face.points_id[0],:]),n/np.linalg.norm(n))
            pts_dists.append(dist)
    pts_dists = np.array(pts_dists)
    #print (face.ID,np.sign(pts_dists))
    if np.all(pts_dists > 0): #any points outside the plane
        tp = 1 #outer
    else:
        tp = -1   # inner
    return tp
                  
def find_loop_type(face_type,loops,pts_list,lines):
    """ check whether the loop is inner loop"""
    points = np.unique([[lines[li].start,lines[li].end] for lp in loops
                        for li in lp.line_list ]) # find all points in the face     
    for lp in loops:
        line_types = []    
        for li in lp.line_list:
            line_vector = lines[li].vector
            cross_products = []
            for p in points: # 检查loop 内其他点是否在线的两侧存在
                if p != lines[li].start and p != lines[li].end:
                    new_vector = pts_list[p,:]- pts_list[lines[li].start,:]
                    cross = np.cross(line_vector,new_vector)
                    cross_products.append(cross)
                    if np.linalg.norm(cross) != 0: #防止三点落一线             
                        norm = cross/np.linalg.norm(cross)
            direction = np.dot(cross_products,norm)
            ln_type = ct.is_same_sign(np.sign(direction)) #np.sign transform vector to -1 0 1
            line_types.append(ln_type)
        line_types = np.array(line_types)
        if np.all(line_types > 0): # all outer or inner lines
            lp.type = 1
        elif np.all(line_types <0):# inner loop
                lp.type = -1 
        else:# hybrid loop
            lp.type = 0      
    

def tri_mesh(facets,faces,lines,coord_array,lines_list_seg,min_length,hole_in_facet):
    tri_array = np.zeros((0,3)) # 储存坐标和三角
    
    for fi in range(len(facets)):         
        #把分割过的线段重新组装
        face_seg,loop2 = ct.line_segements_reorder(fi,facets,faces,lines,lines_list_seg)
        
        ptscoord = coord_array[face_seg, :] # get coordinates of vertices
        
        v = ct.find_normal(ptscoord[0:loop2,:]) # find vectors of plane
        
        ptscoord2 = ct.project_to_plane(ptscoord,v) # get 2d corrdinates, loop2 for 有洞时，normal 不包括洞点，否则normal会偏斜，因为重心
        
        ptscoord3 = list(tuple(i) for i in ptscoord2) #把numpy array 转化为tuple list
        
        plane_dist = np.mean(np.dot(ptscoord[0:loop2,:],v[:,0])) # distance from the origin to the plane
        
        edges = ct.connect_pts(face_seg,loop2) #输入准备,round trip connection of points
        
        if faces[fi].min_length > min_length:
            edge_leng = faces[fi].min_length/3. #set maximal mesh length 
        else:
            edge_leng = min_length #如果此面最小边是圆边    
        
        try: # in case of no hole
            hole_center = hole_in_facet[fi]
        except IndexError:
            hole_center = []
            
        if len(hole_center) > 0: ## get the 2d coo for no empty hole center 
        
            hole_center = list(tuple(i) for i in ct.project_to_plane(hole_center,v)) # use format [(x,y)]  
    
        mesh_pts, mesh_lines = ct.DoTriMesh(ptscoord3,edges,edge_length = edge_leng,holes = hole_center) # create mesh
    
        opn = len(face_seg); offset = coord_array.shape[0] - opn # 点序号 修正准备
            
        mesh_pts = ct.to3d_v2(mesh_pts[opn:,:],v,plane_dist) # convert to 3d coo    
    
        coord_array = np.vstack((coord_array,mesh_pts)) # merge new mesh point
    
        mesh_lines[np.where(mesh_lines>(opn-1))] += offset #序号修正
        
        mesh_lines_temp = np.copy(mesh_lines) # copy，防止loop出错
    
        for p in range(opn): # convert vertices id in new meshto original id
            pind = np.where(mesh_lines_temp==p) 
            mesh_lines[pind] = face_seg[p]
            
        tri_array = np.vstack((tri_array,mesh_lines)) #merge triangles
        faces[fi].mesh_triangles = mesh_lines # save tri to face
        faces[fi].mesh_points = np.unique(mesh_lines) # save pts to face

    return coord_array,tri_array
              
                                
def find_line_attributes(faces,lines,min_length,facets,pts_list):
    """ 一条边属于两个面，找到在哪个面中与短边比值最大,
    并设定此边的分割比例"""
    for li in range(len(lines)):
        lines_length = lines[li].length
        two_face = [];length_two_face = []
        for fi in range(len(faces)): #在所有face中找line位置
            for lp in faces[fi].loops:
                ll = lp.line_list
                if li in ll:
                    two_face.append(fi)
                    length_two_face.append(faces[fi].min_length)
        lines[li].parent_faces = two_face
        min_face = two_face[np.argmin(length_two_face)]
        length_two_face =np.array(length_two_face) 
        if np.all(length_two_face> min_length):# 所属面没有圆线段时
            max_rate = int((lines_length/np.amin(length_two_face))*9+2) #+2避免比例等于0
        else: #有圆线段时
            max_rate = int(lines_length/np.amin(length_two_face))+1
        if lines_length < min_length : # 圆线段不再分割
            max_rate = 2  
        lines[li].seg_rate= max_rate 
        lines[li].max_rate_face = min_face
          
        #find angle
        angle,degree = ct.find_line_angle(facets,two_face,pts_list)
        lines[li].angle = degree
        
        #find line type
        ft1 = faces[two_face[0]].type; ft2 = faces[two_face[1]].type
        if ft1 == ft2 == 1: # only two stock face get a outer line
            lines[li].type = 1
        else:
            lines[li].type = -1
            
    
def find_line_list(lp,lines):
    """"定位loop中有哪些线,返回index list"""
    line_list = []
    edges = ct.round_trip_connect(lp)#connect all points
    for e in edges:
        index = None
        for l in range(len(lines)):
            if (e[0] == lines[l].start and e[1] == lines[l].end)\
            or (e[1] == lines[l].start and e[0] == lines[l].end):
                index = l 
                break
        if index == None:
            raise RuntimeError("错误！找不对应line！loop:%s edge:%s"%(lp,e))
        else:
            line_list.append(index)
    return line_list
                                      
def find_lines(facets,pts_list):
    """"定义所有的线条，唯一，起点号小于终点号"""
    lines_list = []
    for face in facets:
        edges = ct.round_trip_connect(face)#connect all points
        lines_list.extend(edges)
    lines_list =  np.unique(lines_list,axis=0)
    lines = []
    for lid in range(lines_list.shape[0]):
        lines.append(line(lid,lines_list[lid],pts_list))
    return lines

def find_face(pts_list,facets,lines):
    """"定义所有的面"""
    faces = []
    for fid in range(len(facets)):
        new = face(fid,facets[fid],pts_list,lines)
        faces.append(new)
    return faces    
                      
def find_face_norm(face,pts_list):
    p1 = pts_list[face[0]]
    p2 = pts_list[face[1]]
    p3 = pts_list[face[2]]
    v1 = p2 - p1
    v2 = p3 - p2
    norm = np.cross(v1,v2)
    return norm

def find_loop(f,pts_list,lines):
    """"定义所有的loop"""
    lps = []
    if -1 in f: 
        lp = []      
        for v in f:
            if v == -1:
                lps.append(loop(lp,pts_list,lines))
                lp = []
            else:
                lp.append(v)
        if len(lp)>0:
            lps.append(loop(lp,pts_list,lines))
    else:
        lps.append(loop(f,pts_list,lines))
    return lps
    

def find_min_length(loops,lines):
    """找到面内最小边长"""
    ls = []    
    for lo in loops:#collect lengthes       
        for l in lo.line_list:
            ls.append(lines[l].length)
    if len(ls) > 20: #考虑到圆洞
        for l in np.unique(ls):
            minl = l
            if l > 2* np.unique(ls)[0]: #防止圆线段不够精确相等
                break
    else:
        minl = np.amin(ls)        
    return minl
            
if __name__ == "__main__":                    
    w1 = l1 = h1 = 100 
    
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
    pts_list =  np.array(([list(p) for p in pts])) #convert the pts corrdinates array to numpy array
    
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
    
    model = solid_model(pts_list,facets)
    coord_array,tri_array = model.generate_mesh()
    
    from mayavi import mlab
    mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
    mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
        tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
    mlab.show()
    
    
    