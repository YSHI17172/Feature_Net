# -*- coding: utf-8 -*-
import numpy as np
import commontool as ct
import sys

# def timenow(time):
#     import datetime
#     print ('花费时间 %f seconds'%(datetime.datetime.now()-time).seconds )
#     time = datetime.datetime.now()

class solid_model():    
    def __init__(self,pts_list,facets,hole_in_facet=[],min_length=0):
        if len(facets) > 100:
            sys.setrecursionlimit(len(facets)*10)
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

    def generate_mesh(self,uniform_seg=False,unifrom_mesh=False,mesh_length = 0):
        if mesh_length > 0:
            uniform_seg=True;unifrom_mesh=True    
        coord_array,lines_list_seg = ct.line_segmentation_3d(self.pts_list,self.lines,self.faces,self.min_length,uniform_seg,unifrom_mesh,mesh_length)  
        coord_array,tri_array = tri_mesh(self.facets,self.faces,self.lines,coord_array,lines_list_seg,self.min_length,self.hole,unifrom_mesh,mesh_length)
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
                        if fs not in faces_in_feature: 
                            faces_in_feature.append(fs)                                        
                            find_next_adj_face(fs,faces,faces_in_feature)                   
            elif lp.type == 0: #hybrid loop has inner edge
                for ln in lp.line_list:
                    if lines[ln].type == -1: # inner line
                        for fs in lines[ln].parent_faces:
                            if fs not in faces_in_feature:
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

def find_next_adj_face_unidirection(fs,faces,faces_in_feature):   
    """collect adjacent faces recursively, 单向"""
    """非圆洞特征时候，无法使用，因为特征面定义顺序"""
    adj_fs = faces[fs].adjacent_faces[1]
    if adj_fs not in faces_in_feature:
        faces_in_feature.append(adj_fs)
        find_next_adj_face(adj_fs,faces,faces_in_feature)                                                                      
                                                                                            
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
    

def tri_mesh(facets,faces,lines,coord_array,lines_list_seg,min_length,hole_in_facet,uniform_mesh,mesh_length):
    tri_array = np.zeros((0,3)) # 储存坐标和三角
    uniform_length = np.amin([ln.length for ln in lines if ln.length > min_length])
    
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
            if uniform_mesh == True:
                edge_leng = uniform_length/5.
                if mesh_length > 0:
                    edge_leng = mesh_length*2
            else:
                edge_leng = faces[fi].min_length/5. #set maximal mesh length 
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
        if len(two_face) != 2: # error check
            print ('----------------------')
            print ('面定义出错')
            print (two_face)
            print (lines[li].start,lines[li].end)
            for tete in two_face:
                print ('xx',faces[tete-2].points_id)
                print ('xx',faces[tete-1].points_id)
                print ('xx->',faces[tete].points_id)
                print ('xx',faces[tete+1].points_id)
                print ('xx',faces[tete+2].points_id)
                print ('next layer')
            for fi,fc in enumerate(facets):
                if lines[li].start in fc or lines[li].end in fc:
                    print (fc) 
                    for i in range(5):
                        print (facets[fi-i])
                    print ('')
                    for i in range(5):
                        print (facets[fi+i])
                    print ('')
                               
            print ('----------------------')
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
        angle,degree = find_line_angle(faces,two_face,pts_list)
        lines[li].angle = degree
        
        #find line type
        ft1 = faces[two_face[0]].type; ft2 = faces[two_face[1]].type
        if ft1 == ft2 == 1: # only two stock face get a outer line
            lines[li].type = 1
        else:
            lines[li].type = -1

def find_line_angle(faces,two_face,pts_list): 
    norm1 = faces[two_face[0]].norm
    norm2 = faces[two_face[1]].norm   
    angle = np.dot(norm1,norm2)/(np.linalg.norm(norm1) * np.linalg.norm(norm2)) 
    if abs(angle) >1:
        angle = np.sign(angle) # bug fix, sometimes angle > 1
    pts1 = faces[two_face[0]].points_id #face 0 point list
    pts2 = faces[two_face[1]].points_id #face 1 point list
    if pts1 == pts2:
        print ('错误，面定义重复')
        print (pts1,pts2)
        print (two_face)
        print (len(faces))
    #find two points not on the edges
    p1 = pts_list[np.setdiff1d(pts1, pts2)[0]]
    p2 = pts_list[np.setdiff1d(pts2, pts1)[0]]
    convex = np.dot((p2-p1),norm1)
    degree = np.arccos(angle)*180/np.pi

    if convex > 0: #conve edge
        degree+=180
    return angle,degree            
    
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
    norm = 0
    start = 1
    while np.linalg.norm(norm) == 0: # 防止一条直线分两节            
        p1 = pts_list[face[0]]
        p2 = pts_list[face[1]]
        v1 = p2 - p1
        p3 = pts_list[face[start]]
        p4 = pts_list[face[start+1]]
        v2 = p4 - p3
        norm = np.cross(v1,v2) 
        start +=1      
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
    mesh_length = 1
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
    print (cpt_number)
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
        #print (seg_nb,np.pi*2*rad/seg_nb)
        segn_list.append(seg_nb)
        pts.extend((rad * np.cos(angle)+cx, rad * np.sin(angle)+cy,height[h]) # 增加圆洞口点
                    for angle in np.linspace(0, np.pi*2, seg_nb, endpoint=False))
    print (segn_list)
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
            tail = divn%rest
            for sub in range(rest): #分几组，每组n 对 n+1
                for k in range(sub_set):
                    facets.extend([[eptsn+sub*(sub_set+1)+k,
                                    eptsn+prev_divn+sub*sub_set+k,
                                    eptsn+sub*(sub_set+1)+k+1]]) 
                    facets.extend([[eptsn+sub*(sub_set+1)+k+1,
                                    eptsn+prev_divn+sub*sub_set+k,
                                    eptsn+prev_divn+sub*sub_set+k+1]]) 
                #补一个三角
                facets.extend([[eptsn+(sub+1)*(sub_set+1)-1,
                                eptsn+prev_divn+(sub+1)*sub_set,
                                eptsn+(sub+1)*(sub_set+1)]])
            if tail == 0: # 删除最后两个三角
                del facets[-1]
                del facets[-1]
            for t in range(tail-1): #剩余点
                facets.extend([[eptsn+rest*(sub_set+1)+t,
                                eptsn+prev_divn+rest*sub_set+t,
                                eptsn+prev_divn+rest*sub_set+t+1]]) 
                facets.extend([[eptsn+rest*(sub_set+1)+t,
                                eptsn+prev_divn+rest*sub_set+t+1,
                                eptsn+rest*(sub_set+1)+t+1]])
            #头尾
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn+divn-1,eptsn+prev_divn]]) 
            facets.extend([[eptsn+prev_divn-1,eptsn+prev_divn,eptsn]])           

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
    
    model = solid_model(pts_array,facets,hole_facets,min_length=2)
    coord_array,tri_array = model.generate_mesh()
    print ('total feature %d'%len(model.features))
    print ('total faces %d'%len(model.faces))
    print ('feature has %d faces'%len(model.features[0].faces))
    print (model.features[0].faces)
    
    from mayavi import mlab
    mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
    mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
        tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
    mlab.show()
    
    
    