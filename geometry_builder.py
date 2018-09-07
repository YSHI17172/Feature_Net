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
        find_adj_face(self.faces,self.lines,self.pts_list) # update adjacent faces info for every face
        self.features = find_features(self.faces,self.lines) # collect faces to find form feature

    def generate_mesh(self,uniform_seg=False,unifrom_mesh=False,mesh_length = 0):
        if mesh_length > 0:
            uniform_seg=True;unifrom_mesh=True    
        coord_array,lines_list_seg = ct.line_segmentation_3d(self.pts_list,self.lines,self.faces,self.min_length,uniform_seg,unifrom_mesh,mesh_length)  
        for li,ln in enumerate(self.lines): ln.segments = lines_list_seg[li] #update line seg info
        coord_array,tri_array = tri_mesh(self.facets,self.faces,self.lines,coord_array,lines_list_seg,self.min_length,self.hole,unifrom_mesh,mesh_length)
        return coord_array,tri_array
                                                              
class face():
    def __init__(self,fid,f,pts_list,lines):
        self.ID = fid
        self.points_id = f
        self.loops = find_loop(f,pts_list,lines)
        self.min_length = find_min_length(self.loops,lines) 
        self.adjacent_faces = None #list
        self.angle_between_adjacent_face = None #list
        self.norm = ct.find_face_norm(f,pts_list) 
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
        self.concavity = None # 1 convex -1 concave 0 flat

class point():
    def __init__(self,p):
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]

class feature():
    def __init__(self,faces_in_feature,loop,entrance,entrance_type):
        self.faces = faces_in_feature
        self.entrance_faces = entrance
        self.entrance_type = entrance_type
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
        if fc.type == 1: #stock face
            for lp in fc.loops: 
                faces_in_feature = [fc.ID]
                entrance = [fc.ID];entrance_type = [lp.type]
                if lp.type == -1: # inner loop  
                    for ln in lp.line_list:
                        for fs in lines[ln].parent_faces:
                            if fs not in faces_in_feature: 
                                faces_in_feature.append(fs)                                        
                                find_next_inner_adj_face(fs,faces,faces_in_feature,entrance,entrance_type)                   
                elif lp.type == 0: #hybrid loop has inner edge
                    for ln in lp.line_list:
                        if lines[ln].type == -1: # inner line
                            for fs in lines[ln].parent_faces:
                                if fs not in faces_in_feature:
                                    faces_in_feature.append(fs)                                    
                                    find_next_inner_adj_face(fs,faces,faces_in_feature,entrance,entrance_type)
                faces_in_feature = np.setdiff1d(faces_in_feature,fc.ID) # remove base face
                if duplicate_check(faces_in_feature,features) and len(faces_in_feature) > 0:
                    features.append(feature(faces_in_feature,lp,entrance,entrance_type)) # create object 
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
        find_next_adj_face_unidirection(adj_fs,faces,faces_in_feature)                                                                      
                                                                                            
def find_next_inner_adj_face(fs,faces,faces_in_feature,entrance,entrance_type):
    """collect adjacent faces recursively, only add inner face"""
    for adj_fs in faces[fs].adjacent_faces:
        if (adj_fs not in faces_in_feature) and faces[adj_fs].type == -1:
            faces_in_feature.append(adj_fs)
            find_next_inner_adj_face(adj_fs,faces,faces_in_feature,entrance,entrance_type)  
        if (adj_fs not in entrance) and faces[adj_fs].type == 1:
            entrance.append(adj_fs);entrance_type.append(faces[adj_fs].loops[-1].type)
                             
def find_next_adj_face(fs,faces,faces_in_feature):   
    """collect adjacent faces recursively"""
    for adj_fs in faces[fs].adjacent_faces:
        if adj_fs not in faces_in_feature:
            faces_in_feature.append(adj_fs)
            find_next_adj_face(adj_fs,faces,faces_in_feature)                                                      
                                                   
def find_adj_face(faces,lines,pts_list):
    """find adjacent faces for every face"""
    for fa in faces:
        adj_face = []
        adj_angles = []
        for lp in fa.loops:
            for ln in lp.line_list:
                adj_face.extend(np.setdiff1d(lines[ln].parent_faces,fa.ID))
                adj_angles.append(lines[ln].angle)
        fa.adjacent_faces = adj_face 
        fa.angle_between_adjacent_face = adj_angles
                                    
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
    n = face.norm
    n = n/np.linalg.norm(n)
    
    pts_dists = np.dot((pts_list-pts_list[face.points_id[0],:]),n)
    fctype = ct.is_same_sign(np.sign(pts_dists))
    return fctype
                  
def find_loop_type(face_type,loops,pts_list,lines):
    """ check whether the loop is inner loop"""
    if face_type == -1: #inner face
        if len(loops) == 1: # only one loop
            loops[0].type = 1
        else: # more than one loop
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
                if np.all(line_types <0):# inner loop
                    lp.type = -1
                else: 
                    lp.type = 1 
 
    else: #outter face
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
        angle,degree,convex = find_line_angle(faces,lines[li],pts_list)
        lines[li].angle = degree
        lines[li].concavity = convex
        
        #find line type
        ft1 = faces[two_face[0]].type; ft2 = faces[two_face[1]].type
        if ft1 == ft2 == 1: # only two stock face get a outer line
            lines[li].type = 1
        else:
            lines[li].type = -1

def find_line_angle(faces,line,pts_list): 
    two_face = line.parent_faces
    norm1 = faces[two_face[0]].norm
    norm2 = faces[two_face[1]].norm   
    angle = np.dot(norm1,norm2)/(np.linalg.norm(norm1) * np.linalg.norm(norm2)) 
    if abs(angle) >1:
        angle = np.sign(angle) # bug fix, sometimes angle > 1
    degree = np.arccos(angle)*180/np.pi    
        
    pts1 = faces[two_face[0]].points_id #face 0 point list
    pts2 = faces[two_face[1]].points_id #face 1 point list
    if pts1 == pts2:
        print ('错误，面定义重复')
        print (pts1,pts2)
        print (two_face)
        print (len(faces))
    # #find two points not on the edges
    # p1 = pts_list[np.setdiff1d(pts1, pts2)[0]]
    # p2 = pts_list[np.setdiff1d(pts2, pts1)[0]]
    # convex = np.dot((p2-p1),norm1)
    
    #平面不规则相切时候，可能无法正确判断 concavity
    #解决方法：在 line 上选一点，为line 方向上此面最远点。然后投影此点的另一条线到另外一个面的法线方向
    projection_on_face2 = np.dot(pts_list[pts1],norm2)
    projection_sign = np.sign(projection_on_face2 - np.dot(pts_list[pts2],norm2)[0])
    if np.unique(projection_sign).size > 2:# -1,0,1,face 2 切割了 face 1
        pts3 = pts2;pts4 = pts1;convex_norm = norm1 #在face2中选点
    else:
        pts3 = pts1;pts4 = pts2;convex_norm = norm2#在face1中选点
    start = pts_list[line.start] ; end = pts_list[line.end]
    line_vector = line.vector/np.linalg.norm(line.vector)
    projection_on_edge = np.dot(pts_list[pts3],line_vector)
    if np.amax(projection_on_edge) == np.dot(end,line_vector): #end为原 edge 点
        p1 = end
        p1_idx = pts3.index(line.end)
    else:
        p1 = start
        p1_idx = pts3.index(line.start)
    if pts3[p1_idx-1] not in pts4:
        p2_idx = pts3[p1_idx-1]
    else:
        if p1_idx == len(pts3) - 1: #p1正好是loop list 最后一个点
            p2_idx = pts3[0]
        else:
            p2_idx = pts3[p1_idx+1]
    p2 = pts_list[p2_idx]
    convex = np.dot((p2-p1),convex_norm)
    
    # if 7 in two_face:
    #     print(two_face,p1,p2,convex)

    if convex < 0: #concave edge
        degree+=180
    return angle,degree,np.sign(convex)     
    
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
    from read_STEP import read_STEP,closed_shell,get_facets 
    hole_facets = []
    data = read_STEP('3D/'+'blind_hole&through_hole_0.step')   
    step_model = closed_shell(data)
    pts_array,facets,hole_facets = get_facets(step_model,mesh_length=1)
    model = solid_model(pts_array,facets,hole_facets,min_length=1.5)
    #coord_array,tri_array = model.generate_mesh()
    print ('total feature %d'%len(model.features))
    print ('total faces %d'%len(model.faces))

#     for i in range(len(model.features)):
#         print (len(model.features[i].faces))
#         print (model.features[i].faces)
#         print ('')
# 
#     for face in model.faces:
#         print(face.ID)
#         print('face type',face.type)
#         for loop in face.loops:
#             print(loop.line_list)
#             print('type',loop.type)
#         print('')
#     
    # from mayavi import mlab
    # mlab.figure(figure="Mesh", bgcolor = (1,1,1), fgcolor = (0,0,0))
    # mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], \
    #     tri_array,representation='wireframe', color=(0, 0, 0), opacity=0.5)
    # mlab.show()
    # 
    # 
    