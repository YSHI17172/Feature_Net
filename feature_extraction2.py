# -*- coding: utf-8 -*-
import numpy as np
import geometry_builder as gb
import commontool as ct
import itertools

class feature_model():
    def __init__(self,step_model,my_model):
        self.coo_array = step_model.coo_array 
        self.facets = step_model.facets
        self.face_id_dictionary = connect_face_id_between_model(my_model,step_model)
        self.entrance_faces = np.unique([self.face_id_dictionary[face] for face in my_model.features[0].entrance_faces])
        self.face_id_in_intersecting_feature = get_faces_in_intersecting_feature(self,my_model)    
        self.faces = [surface_in_feature(self,face,my_model) for face in step_model.faces]   
        self.features = feature_extraction(self,step_model,my_model)   
        #split_base_plane(self)

def feature_extraction(solid,step_model,my_model): 
    #先找base palne 
    
    step_faces_id_in_feature = solid.face_id_in_intersecting_feature[:]
    decomposed_features = []
    while len(step_faces_id_in_feature) > 0:
        feature_face = step_faces_id_in_feature[0]
        if solid.faces[feature_face].type == 'CYLINDRICAL_SURFACE':
            current_feature = hole_feature(solid,feature_face,step_faces_id_in_feature)
            decomposed_features.append(current_feature)
        else: #Plane
            face_feature_character = solid.faces[feature_face].feature_character
            if solid.entrance_faces.size <3: # only 1 or 2 entrance 说明没有step and slot
                #entrance_normal =  solid.faces[solid.entrance_faces[0]].normal
                entrance_normal = np.array([0,0,1])
                ##face_feature_character 1,2,3,4 都可能是 base plane，4可能被分裂的，需要组合
                #if face_feature_character > 0 \
                if np.allclose(solid.faces[feature_face].normal,entrance_normal) \
                    and solid.faces[feature_face].stock_face == -1:
                        current_feature = pocket_feature(solid,feature_face,step_faces_id_in_feature,entrance_normal)
                        if len(current_feature.faces) > 2:
                            decomposed_features.append(current_feature)
            else:# slot or step
                entrance_normal = np.array([0,0,1])
                #if face_feature_character > 0 \
                if np.allclose(solid.faces[feature_face].normal,entrance_normal) \
                    and solid.faces[feature_face].stock_face == -1\
                    and step_or_slot_check(solid,feature_face) == 'SLOT':                        
                        current_feature = slot_feature(solid,feature_face,step_faces_id_in_feature,entrance_normal)
                        if len(current_feature.faces) > 2:
                            decomposed_features.append(current_feature)
                
                elif np.allclose(solid.faces[feature_face].normal,entrance_normal) \
                and solid.faces[feature_face].stock_face == -1\
                and step_or_slot_check(solid,feature_face)=='STEP':                    
                    current_feature = step_feature(solid,feature_face,step_faces_id_in_feature,entrance_normal)
                    if len(current_feature.faces) > 1:
                        decomposed_features.append(current_feature)
                #pocket adj_face 没有stock face
                elif np.allclose(solid.faces[feature_face].normal,entrance_normal) \
                    and solid.faces[feature_face].stock_face == -1\
                    and step_or_slot_check(solid,feature_face) == 'POCKET' :
                    current_feature = pocket_feature(solid,feature_face,step_faces_id_in_feature,entrance_normal)
                    if len(current_feature.faces) > 2:
                        decomposed_features.append(current_feature) 
              
            step_faces_id_in_feature.remove(feature_face) 
    if len(decomposed_features) > 2 or \
    (len(decomposed_features) == 2 and decomposed_features[0].type == decomposed_features[1].type):
        decomposed_features =  merge_feature(decomposed_features,solid) 
    return decomposed_features

def step_or_slot_check(solid,feature_face):
    """step feature base plane 有至少2个stock face 邻居，而且他们垂直"""
    adj_faces = solid.faces[feature_face].adjacent_faces
    for a, b in itertools.combinations(adj_faces, 2):
        if solid.faces[a].stock_face == solid.faces[b].stock_face == 1\
        and ct.isOrthogonal(solid.faces[a].normal,solid.faces[b].normal):
            return 'STEP'
    else:
        if 1 in [solid.faces[adj_face].stock_face for adj_face in adj_faces]:
            return 'SLOT'
        else:
            return 'POCKET'
            
def merge_feature(features,solid):
    """合并被分裂的 feature """
    feature_type = [feature.type for feature in features]
    feature_set_index = [[i for i,x in enumerate(feature_type) if x == f] for f in set(feature_type)]
    new_features = []
    for feature_set in feature_set_index:
        if len(feature_set) > 1:
            merge_temp = feature_set[:]
            while len(merge_temp) > 0:
                f1 = features[merge_temp[0]]
                for f2_index in merge_temp[1:]:
                    f2 = features[f2_index]
                    if is_same_feature(f1,f2,solid):
                        f1.faces = list(np.unique(f1.faces + f2.faces))
                        f1.base_plane = list(np.unique(f1.base_plane + f2.base_plane))
                        f1.walls = list(np.unique(f1.walls + f2.walls))
                        merge_temp.remove(f2_index)
                        break
                else:
                    new_features.append(f1)
                    merge_temp.pop(0)
        else:
            new_features.append(features[feature_set[0]])             
    return new_features

def is_same_feature(f1,f2,solid):
    if f1.type == 'BLIND' or f1.type == 'THROUGH':
        return is_same_hole(f1,f2,solid)
    elif f1.type == 'POCKET':
        return is_same_pocket(f1,f2,solid)
    elif f1.type == 'SLOT':
        return is_same_slot(f1,f2,solid)
    elif f1.type == 'STEP':
        return is_same_step(f1,f2,solid)
    else:
        raise TypeError('合并特征失败，未知特征种类%s和%s'%(f1.type,f2.type))

def is_same_hole(h1,h2,solid):
    if  np.allclose(h1.support_point,h2.support_point)\
        and np.allclose(h1.normal,h1.normal)\
        and h1.radius == h2.radius:
        return True
    else:
        return False                                                                                             

def is_same_pocket(p1,p2,solid):
    wall1_list = [f for f in p1.faces if f not in p1.base_plane]
    wall2_list = [f for f in p2.faces if f not in p2.base_plane]
    same_counter = 0
    if is_same_plane(solid.faces[p1.base_plane[0]],solid.faces[p2.base_plane[0]]): #same base
        for w1 in wall1_list:
            for w2 in wall2_list:
                if is_same_plane(solid.faces[w1],solid.faces[w2]):
                    same_counter += 1
    if same_counter > 1: #有2个墙是一样的
        return True
    else:
        return False

def is_same_slot(p1,p2,solid):
    wall1_list = [f for f in p1.faces if f not in p1.base_plane]
    wall2_list = [f for f in p2.faces if f not in p2.base_plane]
    same_counter = 0
    if is_same_plane(solid.faces[p1.base_plane[0]],solid.faces[p2.base_plane[0]]): #same base
        for w1 in wall1_list:
            for w2 in wall2_list:
                if is_same_plane(solid.faces[w1],solid.faces[w2]):
                    same_counter += 1
    if same_counter > 1: #有2个墙是一样的
        return True
    else:
        return False
        
def is_same_step(p1,p2,solid):
    wall1_list = [f for f in p1.faces if f not in p1.base_plane]
    wall2_list = [f for f in p2.faces if f not in p2.base_plane]
    same_counter = 0
    if is_same_plane(solid.faces[p1.base_plane[0]],solid.faces[p2.base_plane[0]]): #same base
        for w1 in wall1_list:
            for w2 in wall2_list:
                if is_same_plane(solid.faces[w1],solid.faces[w2]):
                    same_counter += 1
    if same_counter > 0: #有2个墙是一样的
        return True
    else:
        return False
               
def is_same_plane(key_face,second_face):
    # normal 一样，normal 上投影位置位置一样
    if np.allclose(key_face.normal,second_face.normal) \
        and np.dot(key_face.support_point,key_face.normal) == \
        np.dot(second_face.support_point,second_face.normal):
        return True
    else:
        return False  

class pocket_feature():
    def __init__(self,solid,key_face,step_faces_id_in_feature,entrance_normal):
        self.type ='POCKET' 
        self.faces = [key_face]   
        self.base_plane = [key_face]
        self.walls = []
        for adj_index,adj_face in enumerate(solid.faces[key_face].adjacent_faces):            
            if adj_face in solid.face_id_in_intersecting_feature \
            and solid.faces[adj_face].type == 'PLANE' \
            and -1 < solid.faces[adj_face].feature_character <= 1\
            and ct.isOrthogonal(solid.faces[adj_face].normal,entrance_normal)\
            and solid.faces[key_face].angle_between_adjacent_faces[adj_index] < 180: # 底面顺时针转到墙
                self.faces.append(adj_face) ;self.walls.append(adj_face)

class slot_feature():
    def __init__(self,solid,key_face,step_faces_id_in_feature,entrance_normal):
        self.type ='SLOT' 
        self.faces = [key_face]   
        self.base_plane = [key_face]
        self.walls = []
        self.width,self.width_direction = find_slot_width(key_face,solid)
        for adj_index,adj_face in enumerate(solid.faces[key_face].adjacent_faces):            
            if adj_face in solid.face_id_in_intersecting_feature \
            and solid.faces[adj_face].type == 'PLANE' \
            and ct.isOrthogonal(solid.faces[adj_face].normal,entrance_normal)\
            and solid.faces[key_face].angle_between_adjacent_faces[adj_index] < 180\
            and -1 < solid.faces[adj_face].feature_character <= 1: # 底面顺时针转到墙
                self.faces.append(adj_face);self.walls.append(adj_face)
            elif  adj_face in solid.face_id_in_intersecting_feature \
            and solid.faces[adj_face].type == 'PLANE' \
            and ct.isParallel(solid.faces[adj_face].normal,entrance_normal)\
            and face_width_check(solid,adj_face,self.width_direction,self.width):
                self.faces.append(adj_face);self.base_plane.append(adj_face)

class step_feature():
    def __init__(self,solid,key_face,step_faces_id_in_feature,entrance_normal):
        self.type ='STEP' 
        self.faces = [key_face]   
        self.base_plane = [key_face]
        self.walls = []
        self.width,self.width_direction = find_slot_width(key_face,solid)
        for adj_index,adj_face in enumerate(solid.faces[key_face].adjacent_faces):            
            if adj_face in solid.face_id_in_intersecting_feature \
            and solid.faces[adj_face].type == 'PLANE' \
            and ct.isOrthogonal(solid.faces[adj_face].normal,entrance_normal)\
            and solid.faces[key_face].angle_between_adjacent_faces[adj_index] < 180\
            and -1 < solid.faces[adj_face].feature_character <= 1: # 底面顺时针转到墙
                self.faces.append(adj_face);self.walls.append(adj_face)
            elif  adj_face in solid.face_id_in_intersecting_feature \
            and solid.faces[adj_face].type == 'PLANE' \
            and ct.isParallel(solid.faces[adj_face].normal,entrance_normal)\
            and face_width_check(solid,adj_face,self.width_direction,self.width):
                self.faces.append(adj_face);self.base_plane.append(adj_face)

def face_width_check(solid,adj_face,normal,width):
    projection = np.dot(solid.coo_array[solid.faces[adj_face].vertex_list],normal)
    length = abs(np.amax(projection) - np.amin(projection))
    if abs(length-width)<1e-10:
        return True
    else:
        return False
    
def find_slot_width(key_face,solid):
    #find entrance edge
    for edge in solid.faces[key_face].loop_list[0].edge_list:
        if edge.type == 'LINE':
            if 1 in [solid.faces[face].stock_face for face in edge.parent_faces]:
                entrance_edge = edge.geometry.length
                return entrance_edge,edge.geometry.vector
    else:
        raise TypeError('没有找到SLOT底面%s对应入口边宽度'%key_face) 
                                   
                                                                                                                                             
class hole_feature():
    def __init__(self,solid,key_face,step_faces_id_in_feature):
        self.type = solid.faces[key_face].feature_character   
        self.faces = [key_face]   
        self.base_plane = []
        self.walls = [key_face]
        self.support_point = solid.faces[key_face].support_point   
        self.normal = solid.faces[key_face].normal   
        self.radius = solid.faces[key_face].radius   
        step_faces_id_in_feature.remove(key_face)# 从待选去掉此圆柱面
        for adj_face in solid.faces[key_face].adjacent_faces:
            if adj_face in step_faces_id_in_feature and solid.faces[adj_face].type == 'CYLINDRICAL_SURFACE':
                if is_same_surface(solid.faces[key_face],solid.faces[adj_face]):
                    self.faces.append(adj_face);self.walls.append(adj_face)
                    step_faces_id_in_feature.remove(adj_face)  # 组合相同面
        # find the base plane for blind hole
        if self.type == 'BLIND':
            circles = [circle for face in self.faces for circle in solid.faces[face].circle_list]
            position_order = np.argsort([circle.position for circle in circles])  
            bottom_circles = [circle for circle in circles if circle.position == circles[position_order[0]].position]

            for adj_face in solid.faces[key_face].adjacent_faces:
                if adj_face in solid.face_id_in_intersecting_feature and solid.faces[adj_face].type == 'PLANE':   
                    if np.allclose(solid.faces[adj_face].normal,solid.faces[key_face].normal) \
                    and np.dot(solid.faces[adj_face].support_point,solid.faces[key_face].normal) == bottom_circles[0].position:# base plane
                        self.faces.append(adj_face)
                        self.base_plane.append(adj_face)
                                    
class surface_in_feature():
    def __init__(self,solid,step_face,my_model):
        self.ID = step_face.ID      
        self.type = step_face.type
        self.support_point = np.array([step_face.face_geometry.support_point.x,step_face.face_geometry.support_point.y,step_face.face_geometry.support_point.z])
        self.normal = np.array([step_face.face_geometry.normal.x,step_face.face_geometry.normal.y,step_face.face_geometry.normal.z]) 
        self.reference_direction = np.array([step_face.face_geometry.reference_direction.x,step_face.face_geometry.reference_direction.y,step_face.face_geometry.reference_direction.z]) 
        self.loop_list = [boundary_loop(solid,my_model,loop,self.normal) for loop in step_face.bounds] # list of loop objects
        self.vertex_list = find_vertex_list_in_face(self)
        self.stock_face = step_face.stock_face
        self.adjacent_faces,self.angle_between_adjacent_faces= find_adj_faces(self,my_model,solid.face_id_dictionary) # id in step_model
        self.circle_list = [edge for loop in self.loop_list for edge in loop.edge_list if edge.type == 'CIRCLE']
        self.line_list = [edge for loop in self.loop_list for edge in loop.edge_list if edge.type == 'LINE']
        self.feature_character = find_surface_character(self)
        if self.type == 'CYLINDRICAL_SURFACE':self.radius = step_face.face_geometry.radius

def find_vertex_list_in_face(face):
    vertex_list = []
    for loop in face.loop_list:
         for edge in loop.edge_list:
             vertex_list.extend(edge.vertex_index)
    return list(np.unique(vertex_list))

def find_surface_character(self):
    if self.type == 'CYLINDRICAL_SURFACE':
        position_order = np.argsort([circle.position for circle in self.circle_list])  
        bottom_circle = self.circle_list[position_order[0]]
        top_circle = self.circle_list[position_order[-1]]
        seam_line_type = []

        if  1 in seam_line_type:
            face_character = 'BLIND'
        else:
            face_character = 'THROUGH'
        
    elif self.type == 'PLANE':
        face_character = 'PLANE'
        """
        stock face should be [0,0,0,0],base face should be [0,0,0,0]
        inner loop of entrance face could be broken by intersection line
        if all lines are directed clockwisely, the added number of a loop is 0
        """

    return face_character

def base_pattern(alist): 
    for i in range(len(alist)):
        if alist[i] == alist[i-1] == 0: #两个0相连
            return True
    if 2 in alist:
        return True
    else:
        return False
    # alist_merge_circle = []       
    # for i in range(len(alist)):
    #     if alist[i] == alist[i-1] == 2: #两个2相连,合并
    #         continue
    #     else:
    #         alist_merge_circle.append(alist[i])
    #         
    # if alist_merge_circle  == [0,1,2,1] or alist_merge_circle  == [1,0,1,2]\
    # or alist_merge_circle  == [1,2,1,0] or alist_merge_circle  == [2,1,0,1]: #2 为圆弧，02相对
    #     return True
    # elif alist_merge_circle == [0,2] or alist_merge_circle == [2,0]:
    #     return True
    # elif line_characters == [2,1,1] or line_characters == [1,1,2] or line_characters == [1,2,1]:
    #     return True
    # else:
    #     return False
        
    
class boundary_loop():
    def __init__(self,solid,my_model,step_loop,normal):
        self.edge_list = find_edge_list(solid,my_model,step_loop,normal)   #list of edge
        self.type = None # 1,outer,-1,inner,0,hybrid
        self.concavity = None # 1 convex, -1 concave, 0 hybrid, 2 transitional

def find_edge_list(solid,my_model,loop,face_normal):
    edge_list = []
    for oriented_edge in loop.edge_list:
        edge_curve = edge_in_feature(solid,my_model,oriented_edge.element,face_normal)
        edge_list.append(edge_curve)
    return edge_list   
    
class edge_in_feature():
    def __init__(self,solid,my_model,edge_curve,face_normal):
        self.start = np.array([edge_curve.start.x,edge_curve.start.y,edge_curve.start.z])
        self.end = np.array([edge_curve.end.x,edge_curve.end.y,edge_curve.end.z])
        self.vertex_index = edge_curve.vertex_index
        self.type = edge_curve.type
        self.simulation_points = edge_curve.simulation_points # list of points index in coo_array to simulate an arc
        
        if edge_curve.type == 'CIRCLE':
            self.geometry = circle_in_feature(edge_curve.geometry.curve_3d)
            self.position = np.dot(self.geometry.center,face_normal)# 投影在圆柱面上的位置     

        elif edge_curve.type == 'LINE':
            self.geometry = line_in_feature(self,edge_curve.geometry.curve_3d)
            self.position = find_line_position(self.geometry,face_normal)   #1 和 face direction 相同，-1相反，0垂直

            self.simulation_points,self.parent_faces = find_line_segements(self,my_model.lines,solid)
   
def find_line_segements(line,line_list,solid):
    for ln in line_list:
        if set([ln.start,ln.end]) == set(line.vertex_index):
            return ln.segments,[solid.face_id_dictionary[face] for face in ln.parent_faces]           
    else:
        raise NameError('没有找到对应边%s'%ln.vertex_index)
        
class circle_in_feature():
    def __init__(self,step_circle):
        self.radius = step_circle.radius
        self.center = np.array([step_circle.center.x,step_circle.center.y,step_circle.center.z])
        self.normal = np.array([step_circle.normal.x,step_circle.normal.y,step_circle.normal.z])
        self.reference_direction = np.array([step_circle.reference_direction.x,step_circle.reference_direction.y,step_circle.reference_direction.z])
                
class line_in_feature():
    def __init__(self,edge,step_line):
        self.start = np.array([step_line.start.x,step_line.start.y,step_line.start.z])
        self.length = np.linalg.norm(edge.end-edge.start)
        self.vector = np.array([step_line.vector.x,step_line.vector.y,step_line.vector.z])
 
def find_line_position(line,normal):
    if ct.isParallel(line.vector,normal):
        line_position = 1
    elif ct.isOrthogonal(line.vector,normal):
        line_position = 0
    elif ct.isAntiParallel(line.vector,normal):
        line_position = -1
    else:
        raise NameError("检查LINE的定义")      
    return line_position
    


def find_adj_faces(self,my_model,face_id_dict):
    if type(self.ID) == list:
        adj_faces,adj_face_index = np.unique([face for sub_face_id in self.ID for face in my_model.faces[sub_face_id].adjacent_faces if face not in self.ID],return_index=True)
        total_angles = np.array([my_model.faces[sub_face_id].angle_between_adjacent_face[face_index] for sub_face_id in self.ID for face_index,face in enumerate(my_model.faces[sub_face_id].adjacent_faces) if face not in self.ID])
        angle_between_adjacent_faces = total_angles[adj_face_index]
    else:
        adj_faces = my_model.faces[self.ID].adjacent_faces
        angle_between_adjacent_faces = np.array(my_model.faces[self.ID].angle_between_adjacent_face)
    
    step_face_id_list,step_face_index = np.unique([face_id_dict[face_id] for face_id in adj_faces],return_index=True)
    return step_face_id_list,angle_between_adjacent_faces[step_face_index]

def connect_face_id_between_model(model,step_model):
    # connect the face id between two model
    face_id_dict = np.zeros(len(model.faces)).astype(int)
    for i,fc in enumerate(step_model.faces):
        if type(fc.ID) == list:
            for d in fc.ID:
                face_id_dict[d] = i
        else:
            face_id_dict[fc.ID] = i
    return face_id_dict

def get_faces_in_intersecting_feature(self,my_model):
    if len(my_model.features) == 1:
        faces_in_feature = my_model.features[0].faces
    else:
        raise NameError('Feature 数量 %d,请检查!'%(len(my_model.features)))
    
    step_faces_id_in_feature = np.unique([self.face_id_dictionary[face] for face in faces_in_feature])
    return list(step_faces_id_in_feature)

def is_same_surface(key_face,second_face):
    if np.allclose(key_face.support_point,second_face.support_point)\
        and np.allclose(key_face.normal,second_face.normal) \
        and key_face.feature_character == second_face.feature_character:
        return True
    else:
        return False    

if __name__ == "__main__": 
    from read_STEP import read_STEP,closed_shell
    import generate_hks
    import persistence
    import os
    import sys
    os.chdir(sys.path[0])
    
    fname = 676
    
    # feature2 = 'pocket1_pocket2'
    feature1 = 'through_hole_blind_hole'
    feature2 = 'through_hole_blind_hole'
    
    #step_path = 'STEP/%s/%s_%d.step'%(feature1,feature2,fname)
    
    mesh_path = 'input_mesh/intersecting/%s/%s_%d.npz'%(feature1,feature2,fname)
    data = np.load(mesh_path)
    tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
    step_model = data['step_model'][()];my_model = data['my_model'][()]
    data.close()
    
    #step_path = 'STEP/%s/%s_%d.step' %(feature1,feature2,fname)
    #step_path = '3D/%s_%d.step' %(feature2,fname)
    # step_data = read_STEP(step_path) 
    # step_model = closed_shell(step_data)
    # coo_array,facets,hole_facets = step_model.get_facets()
    # my_model = gb.solid_model(coo_array,facets,hole_facets,min_length=1.5)
    # 
    # coord_array,tri_array = my_model.generate_mesh(mesh_length=1)
    # tri_array=tri_array.astype(np.int)
    # save_path = 'input_mesh/intersecting_research/%s/%s_%d'%(feature2,feature2,fname)
    # np.savez_compressed(save_path, step_model=step_model, my_model=my_model,
    # coord_array=coord_array,tri_array=tri_array)
    # print(len(my_model.faces)) 
    
    # Calculate HKS
    HKS = generate_hks.generate_hks(coord_array, tri_array, 0.001, 1000)
    # Calculate persistence
    hks_persistence = persistence.persistence(HKS)
    np.save('temp/hks',hks_persistence)
    
    hks_persistence = np.load('temp/hks.npy')
    
    #import plot_persistence
    #plot_persistence.plot_persistence(coord_array, tri_array, hks_persistence, 'value')
    
    # get feature info
    my_feature_model = feature_model(step_model,my_model,hks_persistence)
    
    print ('共有%d个feature'%len(my_feature_model.features))
    print (my_feature_model.features)
    for feature in my_feature_model.features:
        print (feature.faces)

