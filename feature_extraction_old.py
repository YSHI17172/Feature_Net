# -*- coding: utf-8 -*-
import numpy as np
import geometry_builder as gb
import commontool as ct

class solid_model():
    def __init__(self,step_model,my_model,hks_persistence):
        self.coo_array = step_model.coo_array 
        self.facets = step_model.facets
        self.hks = hks_persistence
        self.face_id_dictionary = connect_face_id_between_model(my_model,step_model)
        self.face_id_in_intersecting_feature = get_faces_in_intersecting_feature(self,my_model)    
        self.faces = [surface_in_feature(self,face,my_model,step_model.coo_array) for face in step_model.faces]   
        self.features = feature_extraction(self,step_model,my_model)   

def feature_extraction(solid,step_model,my_model): 
    #先找base palne 
    
    step_faces_id_in_feature = solid.face_id_in_intersecting_feature[:]
    decomposed_features = []
    while len(step_faces_id_in_feature) > 0:
        feature_face = step_faces_id_in_feature[0]
        if solid.faces[feature_face].type == 'CYLINDRICAL_SURFACE':
            current_feature = hole_feature(solid,feature_face,step_faces_id_in_feature)
            step_faces_id_in_feature.remove(feature_face)            
            decomposed_features.append(current_feature)
        else:
             step_faces_id_in_feature.remove(feature_face) 
        
    return decomposed_features

class hole_feature():
    def __init__(self,solid,key_face,step_faces_id_in_feature):
        self.type = solid.faces[key_face].feature_character   
        self.faces = [key_face]     
        for adj_face in solid.faces[key_face].adjacent_faces:
                if adj_face in step_faces_id_in_feature and solid.faces[adj_face].type == 'CYLINDRICAL_SURFACE':
                    if is_same_surface(solid.faces[key_face],solid.faces[adj_face]):
                        self.faces.append(adj_face)
                        step_faces_id_in_feature.remove(adj_face) 
        # find the base plane for blind hole
        if self.type == 'BLIND':
            circles = [circle for face in self.faces for circle in solid.faces[face].circle_list]
            position_order = np.argsort([circle.position for circle in circles])  
            bottom_circles = [circle for circle in circles if circle.position == circles[position_order[0]].position]

            for adj_face in solid.faces[key_face].adjacent_faces:
                if adj_face in solid.face_id_in_intersecting_feature and solid.faces[adj_face].type == 'PLANE':   
                    if np.array_equal(solid.faces[adj_face].normal,solid.faces[key_face].normal) \
                    and np.dot(solid.faces[adj_face].support_point,solid.faces[key_face].normal) == bottom_circles[0].position:# base plane
                        self.faces.append(adj_face)
                        #vertices_coo_in_plane = solid.coo_array[solid.facets[solid.faces[adj_face].ID]]
                        #idx = np.argmin(np.abs(solid.coo_array-vertex))                
def split_base_plane():
    circles = [circle for face in self.faces for circle in solid.faces[face].circle_list]
    lines = [line for face in self.faces for line in solid.faces[face].line_list]
    position_order = np.argsort([circle.position for circle in circles])  
    bottom_circles = [circle for circle in circles if circle.position == circles[position_order[0]].position]
    top_circles =  [circle for circle in circles if circle.position == circles[position_order[-1]].position]
    
                                    
class surface_in_feature():
    def __init__(self,solid,step_face,my_model,coo_array):
        self.ID = step_face.ID      
        self.type = step_face.type
        self.support_point = np.array([step_face.face_geometry.support_point.x,step_face.face_geometry.support_point.y,step_face.face_geometry.support_point.z])
        self.normal = np.array([step_face.face_geometry.normal.x,step_face.face_geometry.normal.y,step_face.face_geometry.normal.z]) 
        self.reference_direction = np.array([step_face.face_geometry.reference_direction.x,step_face.face_geometry.reference_direction.y,step_face.face_geometry.reference_direction.z]) 
        self.loop_list = [boundary_loop(loop,self.normal,coo_array,solid.hks) for loop in step_face.bounds] # list of loop objects
        self.stock_face = step_face.stock_face
        self.adjacent_faces = find_adj_faces(self,my_model,solid.face_id_dictionary) # id in step_model
        self.circle_list = [edge for loop in self.loop_list for edge in loop.edge_list if edge.type == 'CIRCLE']
        self.line_list = [edge for loop in self.loop_list for edge in loop.edge_list if edge.type == 'LINE']
        self.feature_character = find_surface_character(self)

def find_surface_character(self):
    if self.type == 'CYLINDRICAL_SURFACE':
        position_order = np.argsort([circle.position for circle in self.circle_list])  
        bottom_circle = self.circle_list[position_order[0]]
        top_circle = self.circle_list[position_order[-1]]
        if abs(np.mean(bottom_circle.hks) - np.mean(top_circle.hks)) / np.mean(top_circle.hks) > 0.1:
            face_character = 'BLIND'
        else:
            face_character = 'THROUGH'
        
    elif self.type == 'PLANE':
        face_character = None
        if len(self.loop_list) > 1:
            pass
        else:
            pass
    return face_character

class boundary_loop():
    def __init__(self,step_loop,normal,coo_array,hks_persistence):
        self.edge_list = find_edge_list(step_loop,normal,coo_array,hks_persistence)   #list of edge
        self.type = None # 1,outer,-1,inner,0,hybrid
        self.concavity = None # 1 convex, -1 concave, 0 hybrid, 2 transitional

def find_edge_list(loop,face_normal,coo_array,hks_persistence):
    edge_list = []
    for oriented_edge in loop.edge_list:
        edge_curve = edge_in_feature(oriented_edge.element,face_normal,coo_array,hks_persistence)

        edge_list.append(edge_curve)
    return edge_list   
    
class edge_in_feature():
    def __init__(self,edge_curve,face_normal,coo_array,hks_persistence):
        self.start = np.array([edge_curve.start.x,edge_curve.start.y,edge_curve.start.z])
        self.end = np.array([edge_curve.end.x,edge_curve.end.y,edge_curve.end.z])
        self.type = edge_curve.type
        self.simulation_points = edge_curve.simulation_points # list of points index in coo_array to simulate an arc
        self.hks = [find_hks_value(self.start,coo_array,hks_persistence),find_hks_value(self.end,coo_array,hks_persistence)]
        if edge_curve.type == 'CIRCLE':
            self.geometry = circle_in_feature(edge_curve.geometry.curve_3d)
            self.position = np.dot(self.geometry.center,face_normal)# 投影在圆柱面上的位置          
        elif edge_curve.type == 'LINE':
            self.geometry = line_in_feature(edge_curve.geometry.curve_3d)
            self.position = find_line_position(self.geometry,face_normal)   #1 和 face direction 相同，-1相反，0垂直
    
class circle_in_feature():
    def __init__(self,step_circle):
        self.radius = step_circle.radius
        self.center = np.array([step_circle.center.x,step_circle.center.y,step_circle.center.z])
        self.normal = np.array([step_circle.normal.x,step_circle.normal.y,step_circle.normal.z])
        self.reference_direction = np.array([step_circle.reference_direction.x,step_circle.reference_direction.y,step_circle.reference_direction.z])
                
class line_in_feature():
    def __init__(self,step_line):
        self.start = np.array([step_line.start.x,step_line.start.y,step_line.start.z])
        self.length = step_line.length
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
    
def find_hks_value(vertex,coo_array,hks_value):
    idx = np.argmin(np.abs(coo_array-vertex)) # find the index of pts in coo
    return hks_value[idx,1]

def find_adj_faces(self,my_model,face_id_dict):
    if type(self.ID) == list:
        adj_faces = np.unique([face for sub_face_id in self.ID for face in my_model.faces[sub_face_id].adjacent_faces if face not in self.ID])
    else:
        adj_faces = my_model.faces[self.ID].adjacent_faces
    
    return np.unique([face_id_dict[face_id] for face_id in adj_faces])

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
    if np.array_equal(key_face.support_point,second_face.support_point)\
        and np.array_equal(key_face.normal,second_face.normal) \
        and key_face.feature_character == second_face.feature_character:
        return True
    else:
        return False    
    
# from read_STEP import read_STEP,closed_shell
# import generate_hks
# import persistence
# 
# feature2 = 'blind_hole1_blind_hole2'
# feature1 = 'blind_hole_blind_hole'
# 
# fname = 0
# 
# mesh_path = 'input_mesh/intersecting/%s/%s_%d.npz'%(feature1,feature2,fname)
# #data = np.load('input_mesh/intersecting_research/blind_hole&through_hole/blind_hole&through_hole_0.npz')
# data = np.load(mesh_path)
# tri_array = data['tri_array'].astype(np.int32) ; coord_array = data['coord_array']
# mesh_model = data['model'][()]
# data.close()
# 
# # Calculate HKS
# #HKS = generate_hks.generate_hks(coord_array, tri_array, 0.001, 1000)
# # Calculate persistence
# #hks_persistence = persistence.persistence(HKS)
# 
# hks_persistence = np.load('temp/hks.npy')
# 
# #step_path = '3D/blind_hole&through_hole_0.step'
# step_path = 'STEP/%s/%s_%d.step'%(feature1,feature2,fname)
# step_data = read_STEP(step_path) 
# step_model = closed_shell(step_data)
# coo_array,facets,hole_facets = step_model.get_facets()
# my_model = gb.solid_model(coo_array,facets,hole_facets,min_length=1.5)
# #print(len(my_model.faces)) 
# 
# 
# # get feature info
# feature_model = solid_model(step_model,my_model,hks_persistence)
# 
# # if __name__ == "__main__": 
# #     import os
#     import sys
#     os.chdir(sys.path[0])