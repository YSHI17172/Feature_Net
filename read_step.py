# -*- coding: utf-8 -*-
import numpy as np
import re
import commontool as ct
from shapely.geometry import Polygon

key_pattern = re.compile(r'#\d+')
xyz_pattern = re.compile(r'(\(-*\d+\.\d*,-*\d+\.\d*,-*\d+\.\d*\)|\(-*\d+\.\d*,-*\d+\.\d*\))')

def read_STEP(path):
    with open(path,'r') as infile:
        content=infile.read()#将所有stl数据读到content中, str
    content = content.splitlines() #split single str into list of lines str
    
    if 'ISO-10303-21' not in content[0]: #check format
        raise TypeError('Input format is not STEP!')
    
    data_start = content.index('DATA;')+1
    content = content[data_start:] #去除head 	
    data_end =  content.index('ENDSEC;')
    content = content[:data_end] #去除tail
    
    data = {}
    
    for line in content[:]:        
        pattern = r'(#\d+) = (\w+\(.+\))(\;)' #split line into NO. content ;
        #pattern = r''
        line_split = re.split(pattern,line) 
        line_split = list(filter(None,line_split)) # remove empty string
        
        pattern2 = r'(\w+)\((.*)\)' # solit attribute name and details
        if len(line_split) == 3 and line_split[-1] ==';':
            key = line_split[0] # get the #00
            detail = re.split(pattern2,line_split[1])
            detail = list(filter(None,detail)) # remove empty string
            data[key] = detail #save 
        else: # incomplete line     
            pattern3 = r'(#\d+) = (\w+\(.*|\(.*)' #incomplete head
            line_split = re.split(pattern3,line) 
            line_split = list(filter(None,line_split)) # remove empty string
            if len(line_split) == 2 and '#' in line_split[0]:
                key = line_split[0] # get the #00
                head2connect = line_split[1]        
            else:
                pattern4 = r'(.+\))(\;)' #incomplete tail
                line_split = re.split(pattern4,line) 
                line_split = list(filter(None,line_split)) # remove empty string
                if len(line_split) == 2 and line_split[-1] ==';':
                    head2connect += line_split[0].strip() #add tail, now it is complete
                    detail = re.split(pattern2,head2connect)
                    detail = list(filter(None,detail)) # remove empty string
                    data[key] = detail #save 
                    del(head2connect) #clear var
                elif 'head2connect' in locals(): # add data middle
                    head2connect += line.strip()
                else:
                    print ('这有个错,请检查%s'%line)
    return data

class closed_shell():    
    def __init__(self,data):
        self.faces = find_face(data) 
        self.coo_array = None
        self.facets = None
        self.hole_in_facets = None
        self.line_list = None
        
    def get_facets(self,mesh_length=1):
        line_key_list,line_type_list,line_vertex_id_list,pts_xyz_list = get_line_list(self,mesh_length)
        coo_array = np.array(pts_xyz_list) #方便加减
        face_list = [];face_normal_list = []
        for fc in self.faces:
            face_center = fc.face_geometry.support_point    
            face_center_xyz = np.array([face_center.x,face_center.y,face_center.z])
            face_normal = fc.face_geometry.normal
            face_normal_xyz = np.array([face_normal.x,face_normal.y,face_normal.z]) 
            face_reference_direction = fc.face_geometry.reference_direction
            face_reference_direction_xyz = np.array([face_reference_direction.x,face_reference_direction.y,face_reference_direction.z]) 
                
            face_bounds = []
            for bidx,bound in enumerate(fc.bounds):#find index of edge in line_list
                face_boudary_edge_list = [];edge_curve_list = []
                if bidx > 0: face_bounds.append(-1)            
                for oriented_edge in bound.edge_list:
                    edge_curve = oriented_edge.element
                    edge_curve_list.append(edge_curve)
                    edge_curve_idx = line_key_list.index(edge_curve.key)
                    face_boudary_edge_list.append(edge_curve_idx)
    
                if fc.type == 'CYLINDRICAL_SURFACE':
                    face_radius = fc.face_geometry.radius
                    # find out circle edge index 
                    circle_edge = [edge_idx for edge_idx in face_boudary_edge_list if line_type_list[edge_idx] == 'CIRCLE']
                    
                    #check if there is complete circle in the face
                    circle_list = [ec for ec in edge_curve_list if ec.type == 'CIRCLE']
                    whole_circle = check_whole_circle(circle_list) 
                        
                    if len(circle_edge) == 2: #圆柱面只有2个圆弧
                        cylindrical_facets = two_arc_into_facets(line_vertex_id_list,circle_edge,face_normal_xyz,coo_array,whole_circle,bound)
        
                    else:
                        cylindrical_lines = split_cylindrical_surface_into_lines(face_radius,face_center_xyz,face_normal_xyz,face_reference_direction_xyz,line_vertex_id_list,circle_edge,coo_array)
                        #print(coo_array[[253, 403, 9, 13,16,254]])
                        #print(cylindrical_lines)
                        cylindrical_facets = pair_cylindrical_lines_into_facets(cylindrical_lines,coo_array,whole_circle)
 
                    fc.ID = [len(face_list)+i for i in range(len(cylindrical_facets))]#把分割面与圆柱面关联
                    face_list.extend(cylindrical_facets)
                    for i in range(len(cylindrical_facets)):face_normal_list.append([face_normal_xyz,face_reference_direction_xyz])
                            
                elif fc.type == 'PLANE':
                    edge_list = [line_vertex_id_list[edge] for edge in face_boudary_edge_list]
                    #print(edge_list)
                    face_bounds.extend(ct.edge_loop2point_loop(edge_list)) #convert line loop to vertex loop
                    fc.stock_face = check_stock_face(face_center_xyz,face_normal_xyz,coo_array) #update stock face information
                
                else:
                    raise NameError("检查面的定义%s"%fc.key)
                bound.edge_index_list = face_boudary_edge_list
            if len(face_bounds) > 0:
                fc.ID=len(face_list)
                face_list.append(face_bounds)
                face_normal_list.append([face_normal_xyz,face_reference_direction_xyz])
    
        #check the hole in the face
        self.hole_in_facets = find_hole(self,coo_array,face_list)          
        #[print (fc) for fc in face_list]
        #调整点序，保证 normal
        face_normals,new_point_loop_list = ct.get_normal_from_model(coo_array,face_list)
        
        #同一平面两个面 touch 并且有共享点不涉及第三面
        self.coo_array,self.facets = check_face_touches(coo_array,new_point_loop_list,face_normal_list,self)
    
        return self.coo_array,self.facets,self.hole_in_facets
    
class advanced_face():
    def __init__(self,data,key):
        self.ID = None
        self.key = key
        self.type = None
        self.bounds = None # loop list
        self.face_geometry = None
        self.stock_face = False

class loop():
    def __init__(self,data,key):
        self.key = key
        self.edge_list = None   #list of oriented_edge
        self.type = None # 1,outer,-1,inner,0,hybrid
        self.concavity = None # 1 convex, -1 concave, 0 hybrid, 2 transitional
        self.edge_index_list = None # lsit of index in line_list

class oriented_edge():
    def __init__(self,data,key):
        self.key = key
        self.element = None   #edge_curve
            
class plane():
    def __init__(self,data,key):
        self.ID = None
        self.key = key
        self.type = None
        self.support_point = None
        self.normal = None
        self.reference_direction = None

class cartesian_point():
    def __init__(self,data,key):
        self.key = key
        self.x = None
        self.y = None
        self.z = None

class direction():
    def __init__(self,data,key):
        self.key = key
        self.x = None
        self.y = None
        self.z = None
        
class cylindrical_surface():
    def __init__(self,data,key):
        self.key = key
        self.radius = None
        self.support_point = None
        self.normal = None
        self.reference_direction = None

class edge_curve():
    def __init__(self,data,key):
        self.key = key
        self.start = None
        self.end = None
        self.geometry = None #surface_curve or seam_curve
        self.type = None
        self.simulation_points = None # list of points index in coo_array to simulate an arc
        self.vertex_index = None # index in coo_array
        
class surface_curve():
    def __init__(self,data,key):
        self.key = key
        self.curve_3d = None # line or circle
        self.associated_geometry = None #LIST OF pcurve_or_surface
        self.type = None
  
class line():
    def __init__(self,data,key):
        self.key = key
        self.start = None
        self.length = None
        self.vector = None

class circle():
    def __init__(self,data,key):
        self.key = key
        self.radius = None
        self.center = None
        self.normal = None
        self.reference_direction = None
        
class pcurve():
    def __init__(self,data,key):
        self.key = key
        self.basis_surface = None
        self.reference_to_curve = None
        
class definitional_representation():
    def __init__(self,data,key):
        self.key = key
        self.items = None
        self.context_of_items = None

class representation_context():
    def __init__(self,data,key):
        self.key = key
        self.identifier = None
        self.type = None

class seam_curve():
    #subtype of surface_curve
    def __init__(self,data,key):
        self.key = key
        self.curve_3d = None
        self.associated_geometry = None
        self.master_representation = None
        

def find_plane(data,face_plane_key):
    plane_def_key = key_pattern.findall(data[face_plane_key][1])[0]
    current_plane = plane(data,face_plane_key)
    if data[plane_def_key][0] == 'AXIS2_PLACEMENT_3D':  
        plane_info_key =  key_pattern.findall(data[plane_def_key][1])
        plane_point_key = plane_info_key[0]
        current_plane.support_point = find_point(data,plane_point_key)
        plane_normal_key = plane_info_key[1]
        current_plane.normal = find_direction(data,plane_normal_key)
        plane_ref_direction_key = plane_info_key[2]
        current_plane.reference_direction = find_direction(data,plane_ref_direction_key)                
    else:
        print ('请检查此PLANE定义%s，找不到AXIS2_PLACEMENT_3D'%(data[plane_def_key]))  
    return current_plane      

def find_vertex(data,info_key):
    if data[info_key][0] == 'VERTEX_POINT': # start and end point
        vertex_point_key = key_pattern.findall(data[info_key][1])[0]
        vertex = find_point(data,vertex_point_key)
    else:
        print ('请检查此VERTEX_POINT定义%s'%(data[info_key])) 
    return vertex
        
def find_point(data,key):
    if data[key][0] == 'CARTESIAN_POINT':           
        point = cartesian_point(data,key)
        xyz = eval(xyz_pattern.findall(data[key][1])[0])
        point.x = xyz[0]
        point.y = xyz[1]
        if len(xyz) > 2:
            point.z = xyz[2]
    else:
        print ('请检查此CARTESIAN_POINT定义%s'%(data[key])) 
    return point

def find_direction(data,key):
    if data[key][0] == 'DIRECTION':           
        vector = direction(data,key)
        xyz = eval(xyz_pattern.findall(data[key][1])[0])
        vector.x = xyz[0]
        vector.y = xyz[1]
        if len(xyz) > 2:
            vector.z = xyz[2]
    else:
        print ('请检查此DIRECTION定义%s'%(data[key])) 
    return vector

def find_cylindrical_surface(data,face_plane_key):
    plane_def_key = key_pattern.findall(data[face_plane_key][1])[0]
    current_surface = cylindrical_surface(data,face_plane_key)
    current_surface.radius = float(re.findall(r',#\d+,(\d+\.\d*)',data[face_plane_key][1])[0])
    if data[plane_def_key][0] == 'AXIS2_PLACEMENT_3D':  
        plane_info_key =  key_pattern.findall(data[plane_def_key][1])
        plane_point_key = plane_info_key[0]
        current_surface.support_point = find_point(data,plane_point_key)
        plane_normal_key = plane_info_key[1]
        current_surface.normal = find_direction(data,plane_normal_key)
        plane_ref_direction_key = plane_info_key[2]
        current_surface.reference_direction = find_direction(data,plane_ref_direction_key)                
    else:
        print ('请检查此CYLINDRICAL_SURFACE定义%s，找不到AXIS2_PLACEMENT_3D'%data[plane_def_key])  
    return current_surface 

def find_face(data):
    face_list = []
    for key in data:
        value = data[key]
        if value[0] == 'CLOSED_SHELL':
            advance_face_keys = key_pattern.findall(value[1]) #get the faces key
            for fck in advance_face_keys: #loop over all faces
                face_value = data[fck]
                if face_value[0] == 'ADVANCED_FACE':
                    current_face = advanced_face(data,fck) # create class to save info
                    face_detail = re.findall(r',\((#\d+.*)\),(#\d+),',face_value[1])
                    face_boundary_keys = face_detail[0][0]                     
                    face_plane_key = face_detail[0][1]
                    current_face.bounds = []     
              
                    if data[face_plane_key][0] == 'PLANE':
                        current_face.type = 'PLANE'
                        current_face.face_geometry = find_plane(data,face_plane_key)                        
                    elif  data[face_plane_key][0] == 'CYLINDRICAL_SURFACE':
                        current_face.type = 'CYLINDRICAL_SURFACE'
                        current_face.face_geometry = find_cylindrical_surface(data,face_plane_key)
                    else:
                        print ('请检查此ADVANCED_FACE定义%s'%face_value)
                        
                    face_boundary_key_list = key_pattern.findall(face_boundary_keys)
                    for face_boundary_key in face_boundary_key_list:
                        if data[face_boundary_key][0] == 'FACE_BOUND':
                            current_face.bounds.append(find_loop(data,face_boundary_key))                           
                        else:
                            print ('请检查此FACE_BOUND定义%s'%(data[face_boundary_key]))
                    face_list.append(current_face)
                else:
                    print ('请检查此ADVANCED_FACE定义%s'%(face_value))
    return face_list

def find_loop(data,face_boundary_key):
    boundary_loop_key = re.findall(r',(#\d+),',data[face_boundary_key][1])[0]
    if data[boundary_loop_key][0] == 'EDGE_LOOP':
        boundary_loop_edge_keys = re.findall(r'(#\d+)',data[boundary_loop_key][1])  
        current_loop = loop(data,boundary_loop_key)       
        current_loop.edge_list = [] 
        #if len(boundary_loop_edge_keys) == 1: #只有一条边
                                          
        for edge_key in boundary_loop_edge_keys:
            current_loop.edge_list.append(find_oriented_edge(data,edge_key))
    else:
        print ('请检查此EDGE_LOOP定义%s'%data[boundary_loop_key] )
    return current_loop

def find_oriented_edge(data,edge_key):
    oriented_edge_info = data[edge_key]
    if oriented_edge_info[0] == 'ORIENTED_EDGE':
        edge_curve_key = key_pattern.findall(oriented_edge_info[1])[0]
        current_oriented_edge = oriented_edge(data,edge_key)  
        current_oriented_edge.element =  find_edge_curve(data,edge_curve_key)                                        
    else:
        print ('请检查此ORIENTED_EDGE定义%s'%(oriented_edge_info))
    return current_oriented_edge

def find_edge_curve(data,edge_curve_key):
    current_edge = edge_curve(data,edge_curve_key)
    if data[edge_curve_key][0] == 'EDGE_CURVE':
        edge_curve_info = key_pattern.findall(data[edge_curve_key][1])
        edge_start_key = edge_curve_info[0]
        current_edge.start = find_vertex(data,edge_start_key)
        edge_end_key = edge_curve_info[1]
        current_edge.end = find_vertex(data,edge_end_key)

        surface_cure_key = edge_curve_info[2]
        if data[surface_cure_key][0] == 'SURFACE_CURVE':  
            current_edge.geometry = find_surface_curve(data,surface_cure_key,current_edge)
        elif data[surface_cure_key][0] == 'SEAM_CURVE':
            current_edge.geometry = find_seam_curve(data,surface_cure_key,current_edge)
    else:
        print ('请检查此EDGE_CURVE定义%s'%(data[edge_curve_key]))
    return current_edge

def find_surface_curve(data,surface_cure_key,current_edge):   
    if data[surface_cure_key][0] == 'SURFACE_CURVE':    
        current_surface_curve = surface_curve(data,surface_cure_key)
                   
        entity_key = re.findall(r',(#\d+),\(.+\),',data[surface_cure_key][1])[0]
        pcurve_keys = key_pattern.findall(re.findall(r',#\d+,\((.+)\),',data[surface_cure_key][1])[0])  
          
        if data[entity_key][0] == 'LINE':
            current_edge.type = 'LINE'
            current_surface_curve.curve_3d = find_line(data,entity_key)
        elif data[entity_key][0] == 'CIRCLE':
            current_edge.type = 'CIRCLE'
            current_surface_curve.curve_3d = find_circle(data,entity_key)  
        else:
            print ('请检查此SURFACE_CURVE定义%s,找不到对应实体'%(data[surface_cure_key]))            
        
        current_surface_curve.associated_geometry = []
        for pcurve_key in pcurve_keys:
            current_surface_curve.associated_geometry.append(find_pcurve(data,pcurve_key))
       
    else:
        raise NameError('请检查此SURFACE_CURVE定义%s'%(data[surface_cure_key]))
    return current_surface_curve

def find_seam_curve(data,seam_cure_key,current_edge):   
    if data[seam_cure_key][0] == 'SEAM_CURVE':    
        current_seam_curve = seam_curve(data,seam_cure_key)
                   
        entity_key = re.findall(r',(#\d+),\(.+\),',data[seam_cure_key][1])[0]
        pcurve_keys = key_pattern.findall(re.findall(r',#\d+,\((.+)\),',data[seam_cure_key][1])[0])  
          
        if data[entity_key][0] == 'LINE':
            current_edge.type = 'LINE'
            current_seam_curve.curve_3d = find_line(data,entity_key)
        elif data[entity_key][0] == 'CIRCLE':
            current_edge.type = 'CIRCLE'
            current_seam_curve.curve_3d = find_circle(data,entity_key)  
        else:
            print ('请检查此SEAM_CURVEE定义%s,找不到对应实体'%(data[seam_cure_key]))            
        
        current_seam_curve.associated_geometry = []
        for pcurve_key in pcurve_keys:
            current_seam_curve.associated_geometry.append(find_pcurve(data,pcurve_key))
        
    else:
        raise NameError('请检查此SEAM_CURVE定义%s'%(data[seam_cure_key]))
    return current_seam_curve

def find_pcurve(data,pcurve_key):
    if data[pcurve_key][0] == 'PCURVE':  
        pcurve_info_keys = key_pattern.findall(data[pcurve_key][1])
        current_pcurve = pcurve(data,pcurve_key)
        basis_surface_key = pcurve_info_keys[0]           
        definition_representation_key = pcurve_info_keys[1]
        
        if data[basis_surface_key][0] == 'PLANE':
            current_pcurve.basis_surface = find_plane(data,basis_surface_key)
        # elif data[basis_surface_key][0] == 'CIRCLE':
        #     pass
        elif data[basis_surface_key][0] == 'CYLINDRICAL_SURFACE':
            current_pcurve.basis_surface = find_cylindrical_surface(data,basis_surface_key)            
        else:
            print ('请检查此PLANE定义%s'%(data[basis_surface_key]))
            
        if data[definition_representation_key][0] == 'DEFINITIONAL_REPRESENTATION':
            current_pcurve.reference_to_curve = find_definitional_representation(data,definition_representation_key)
        else:
            print ('请检查此DEFINITIONAL_REPRESENTATION定义%s'%data[definition_representation_key])   
    else:
        print ('请检查此Pcurve定义%s'%(data[pcurve_key]) )
    return current_pcurve

def find_line(data,line_key):
    line_info_keys = key_pattern.findall(data[line_key][1])
    line_point_key = line_info_keys[0] 
    line_vector_key = line_info_keys[1]
    current_line = line(data,line_key)
    current_line.start = find_point(data,line_point_key)
    if data[line_vector_key][0] == 'VECTOR':        
        current_line.length = float(re.findall(r',#\d+,(\d+\.\d*)',data[line_vector_key][1])[0])  
        vector_direction_key = key_pattern.findall(data[line_vector_key][1])[0]
        current_line.vector = find_direction(data,vector_direction_key)
    else:
        print ('请检查此VECTOR定义%s'%data[line_vector_key])
    return current_line

def find_circle(data,entity_key):   
    current_circle = circle(data,entity_key)
    current_circle.radius = float(re.findall(r',#\d+,(\d+\.\d*)',data[entity_key][1])[0])
    placement_key = key_pattern.findall(data[entity_key][1])[0]
    if data[placement_key][0] == 'AXIS2_PLACEMENT_3D':        
        circle_info_keys = key_pattern.findall(data[placement_key][1])
        circle_center_key = circle_info_keys[0]
        circle_normal_key = circle_info_keys[1]
        circle_refdirection_key = circle_info_keys[2]
        current_circle.center = find_point(data,circle_center_key)
        current_circle.normal = find_direction(data,circle_normal_key)
        current_circle.reference_direction = find_direction(data,circle_refdirection_key)   
    elif data[placement_key][0] == 'AXIS2_PLACEMENT_2D': 
        circle_info_keys = key_pattern.findall(data[placement_key][1])       
        circle_center_key = circle_info_keys[0]
        circle_normal_key = circle_info_keys[1]
        current_circle.center = find_point(data,circle_center_key)
        current_circle.normal = find_direction(data,circle_normal_key)
    else:
        print ('请检查此CIRCLE定义%s，找不到AXIS2_PLACEMENT_3D'%data[entity_key])  
    return current_circle 
    

def find_definitional_representation(data,definition_representation_key):
    current_definitional_representation = definitional_representation(data,definition_representation_key)
    item_key = re.findall(r',\((#\d+)\),#\d+',data[definition_representation_key][1])[0]
    context_key = re.findall(r',\(#\d+\),(#\d+)',data[definition_representation_key][1])[0]                       
    current_definitional_representation.context_of_items = representation_context(data,context_key)# 未处理细节
    if data[item_key][0] == 'LINE':
        current_definitional_representation.items = find_line(data,item_key)
    elif data[item_key][0] == 'B_SPLINE_CURVE_WITH_KNOTS':
        pass
    elif  data[item_key][0] == 'CIRCLE':
        current_definitional_representation.items = find_circle(data,item_key)
    else:
        print ('请检查此%s,找不到对应类定义%s'%(data[definition_representation_key],data[item_key]))
    return current_definitional_representation
            

def find_hole(self,coo_array,facets):
    #定义洞
    hole_facets = []
    for fi,fc in enumerate(facets):
        if -1 in fc:
            cut_point = fc.index(-1)            
            loop1 = fc[:cut_point]
            loop2 = fc[cut_point+1:]
            poly1 = Polygon(coo_array[loop1])
            poly2 = Polygon(coo_array[loop2])
            if poly2.within(poly1): #poly2 是洞
                hole_center = np.mean(coo_array[loop2], axis=0).flatten() # 重心/坐标平均值
            elif poly1.within(poly2): #poly1 是洞
                hole_center = np.mean(coo_array[loop1], axis=0).flatten()
                facets[fi] = loop2 + [-1] + loop1 #调换顺序
                for advfc in self.faces:
                    if advfc.ID == fi:
                        advfc.bounds = [advfc.bounds[1],advfc.bounds[0]]
            else:
                print('loop1 面积 %.2f, loop2 面积 %.2f'%(poly1.area,poly2.area))
                print('loop2 在 loop1 中？',poly2.within(poly1))
                print('loop1 在 loop2 中？',poly1.within(poly2))
                print('loop1 和 loop2 相交？',poly1.intersects(poly2))
                print('loop1 和 loop2 不相干？',poly1.disjoint(poly2))
                print('loop1',poly1.bounds)
                print('loop2',poly1.bounds)
                raise NameError("检查洞定义 %s"%fc)        
            hole_facets.append(hole_center)
        else:
            hole_facets.append([])
    return hole_facets

def check_face_touches(coo_array,facets,face_normal_list,model):
    new_coo = np.copy(coo_array)
    new_facets = facets[:]
    c = 0
    while c != len(new_facets): # 当面的数量不再变化时，停止循环
        c = len(new_facets)
        new_coo, new_facets = merge_face(new_coo,new_facets,face_normal_list,model) 
    return new_coo,new_facets

def get_plane_3d_axis(norm_and_ref):
    norm_direction = norm_and_ref[0]
    reference_direction = norm_and_ref[1]
    third_direction = np.cross(reference_direction,norm_direction)
    return np.array([norm_direction,reference_direction,third_direction]).T
    
def merge_face(coo_array,facets,face_normal_list,model):
    """
    当一个 polygon 包含另一个的所有点的时候，考虑合并他们
    """
    for i,loop in enumerate(facets):
        rest = facets[:]
        rest.remove(loop)
        for other in rest:
            if set(loop) < set(other): #被包含
                other_id = facets.index(other)
                #p1 = Polygon(coo_array[loop])
                #p2 = Polygon(coo_array[other])                
                
                #有时候3d polygon touch check 假false，增加2d投影check
                plane_3d_axis = get_plane_3d_axis(face_normal_list[i])
                p1_2d = Polygon(ct.project_to_plane(coo_array[loop],plane_3d_axis))
                p2_2d = Polygon(ct.project_to_plane(coo_array[other],plane_3d_axis))
                #if p1.touches(p2):
                if p1_2d.touches(p2_2d): 
                    new_coo,new_facets = remove_loop(loop,coo_array,facets,model) #去掉这个小面
                    update_face_in_model(model,other_id,i)
                    update_face_id(model,i)
                    return new_coo,new_facets 
                
    else: #循环走完时候返回
        return coo_array,facets

def update_face_in_model(model,facet_merge,facet_remove):
    """把两面相同边去除，不同边插入"""
    for idx,fc in enumerate(model.faces):
        if fc.ID == facet_remove:
            face_remove = idx
        if fc.ID == facet_merge:
            face_merge = idx

    l1 = model.faces[face_remove].bounds[0].edge_list
    l2 = model.faces[face_merge].bounds[0].edge_list
    remove_list = []
    for i,edge in enumerate(l1):
        for j,compare in enumerate(l2):
            if is_same_edge(edge,compare):
                remove_list.append(j)
                break
        else:
            add = edge
    #复核条件
    if len(remove_list) + 1 == len(l1):
        new_loop = l2[:]
        new_loop[remove_list.pop()] = add
        new_loop = [edge for idx,edge in enumerate(new_loop) if idx not in remove_list]
        model.faces[face_merge].bounds[0].edge_list = new_loop
    else:
        raise NameError('面%d 和面%d 合并失败,插入边不是一条'%(facet_merge,facet_remove))
    model.faces.pop(face_remove)

def is_same_edge(e1,e2):
    if e1.element.type == e2.element.type \
       and e1.element.vertex_index == e2.element.vertex_index\
       and set(e1.element.simulation_points) == set(e2.element.simulation_points):
        return True
    else:
        return False

def remove_loop(loop,coo_array,facets,model):
    new_coo = np.copy(coo_array)
    new_facets = facets[:]
    new_facets.remove(loop) #先去掉这个面
    flattened = [p for loop in facets for p in loop] # facets 压平
    # 只有小面和大面独有的点被去除，以防影响第三面
    to_be_remove = [p for p in loop if flattened.count(p) == 2] 
    for i,p in enumerate(np.unique(to_be_remove)):
        new_coo, new_facets = remove_point(p-i,new_coo,new_facets,model)        
        update_point_id(model,p-i)
    return new_coo,new_facets

def remove_point(p,coo_array,facets,model):
    new_facets = []
    new_coo = np.delete(coo_array,p,0)
    for face_index,loop in enumerate(facets):
        new_loop = []
        for x in loop:
            if x < p: #前面的点号不变
                new_loop.append(x)
            elif x > p: #后面的点号减1
                new_loop.append(x-1) 
        if len(new_loop) > 0: # 空掉的 face去除
            new_facets.append(new_loop)
        else:
            update_face_id(model,face_index)
    return new_coo,new_facets

def update_idx_list(idx,idx_list):
    new_vidx_list =[]
    for vidx in idx_list:
        if vidx < idx: #前面的点号不变
            new_vidx_list.append(vidx)
        elif vidx > idx: #后面的点号减1
            new_vidx_list.append(vidx-1) 
    return new_vidx_list

def update_point_id(model,pid):
    """ 去除冗余面时，更新模型内对应circle point list id """
    for fc in model.faces:
        for loop in fc.bounds:
            for oriented_edge in loop.edge_list:
                edge_curv = oriented_edge.element
                edge_curv.vertex_index = update_idx_list(pid,edge_curv.vertex_index)
                if edge_curv.type == 'CIRCLE':
                    edge_curv.simulation_points = update_idx_list(pid,edge_curv.simulation_points)
    for lnid,line in enumerate(model.line_list):
       model.line_list[lnid] =  update_idx_list(pid,line)
    
def update_face_id(model,face_index):
    """ 去除冗余面时，更新模型内对应面id """
    for fc in model.faces:
        if type(fc.ID) == list:
            new_id = []
            for i in fc.ID:
                if i < face_index: #前面的点号不变
                    new_id.append(i)
                elif i > face_index: #后面的点号减1
                    new_id.append(i-1) 
            fc.ID = new_id
        else:
            if fc.ID > face_index:
                fc.ID -= 1
            elif fc.ID  == face_index:
                fc.ID = None

def get_vertex_list(model):
    pts_xyz_list = []
    pts_id_list = []
    for fc in model.faces: #get the vertices in the model
        for bound in fc.bounds:
            for oriented_edge in bound.edge_list:
                curve_edge = oriented_edge.element
                startxyz = curve_edge.start
                endxyz = curve_edge.end
                v1 = [startxyz.x,startxyz.y,startxyz.z]
                v1_key = startxyz.key
                v2_key = endxyz.key
                v2 = [endxyz.x,endxyz.y,endxyz.z]
                if v1_key not in pts_id_list:
                    pts_id_list.append(v1_key)
                    pts_xyz_list.append(v1)
                if v2_key not in pts_id_list:
                    pts_id_list.append(v2_key)
                    pts_xyz_list.append(v2)
    
    return pts_xyz_list,pts_id_list

def check_split_line_order(split_line,coo_array,face_normal_xyz):
    """ 三点一线时候，保证点序正确"""
    projection_length = np.dot(coo_array[split_line],face_normal_xyz)
    new_split_line = np.array(split_line)[np.argsort(projection_length)]
    
    return list(new_split_line)

def check_orthogonal(facet,coo_array):
    new_facet = [p for l in facet for p in l]
    facet_line_loop = ct.point_loop_to_line_loop(new_facet) #两条线一组，以便检查角度    
    angles_in_facet = []    
    for pair_idx,line_and_line in enumerate(facet_line_loop):
        line1 = coo_array[line_and_line[0][1]] - coo_array[line_and_line[0][0]]
        line2 = coo_array[line_and_line[1][1]] - coo_array[line_and_line[1][0]]
        if ct.isOrthogonal(line1,line2):
            angles_in_facet.append(True) 
        else: #防止一边三点
            pre_line = facet_line_loop[pair_idx-1]
            line0 = coo_array[pre_line[0][1]] - coo_array[pre_line[0][0]]
            if ct.isParallel(line1,line2,1e-15):
                pass
            elif ct.isOrthogonal(line0,line2) and ct.isOrthogonal(line0,line1):# 虽然line1 line2 在3d space 不平行，但是在2d 面内平行
                pass
            else:
                angles_in_facet.append(False) 
                
                
    return angles_in_facet               

def get_line_list(model,mesh_length):
    pts_xyz_list,pts_id_list = get_vertex_list(model)

    line_vertex_id_list = []
    line_key_list = []
    line_type_list = []  

    for fc in model.faces: #get edge list
        for bound in fc.bounds:
            for oriented_edge in bound.edge_list:                
                edge_curve = oriented_edge.element
                startxyz = edge_curve.start
                endxyz = edge_curve.end
                v1_key = startxyz.key
                v2_key = endxyz.key
                start_idx = pts_id_list.index(v1_key)
                end_idx = pts_id_list.index(v2_key)
                edge_curve.vertex_index = [start_idx,end_idx]
                if edge_curve.key not in line_key_list:
                    line_key_list.append(edge_curve.key)
                    line_type_list.append(edge_curve.type)
                    
                    if edge_curve.type == 'CIRCLE':            
                        circle = edge_curve.geometry.curve_3d
                        rad = circle.radius
                        cent = circle.center 
                        norm = circle.normal
                        reference_direction = circle.reference_direction
                        centxyz = np.array([cent.x,cent.y,cent.z])
                        normxyz = np.array([norm.x,norm.y,norm.z])
                        reference_direction_xyz = np.array([reference_direction.x,reference_direction.y,reference_direction.z])
                        circle_start = np.array([startxyz.x,startxyz.y,startxyz.z])     
                        circle_end = np.array([endxyz.x,endxyz.y,endxyz.z])
                        v1 = circle_start - centxyz
                        v2 = circle_end - centxyz
                        pts_number = len(pts_xyz_list) #已有点数
                        cpt_number = int(2*np.pi*rad/mesh_length)+1 #圆模拟点数
                        start_vector =  reference_direction_xyz*rad#假设圆圈起始点，以便同一圆柱面不同圆弧分段一致
                        whole_circle_points = [list(np.dot(ct.rotation_matrix(normxyz,rotate),start_vector) + centxyz) # 插入点到圆弧
                                        for rotate in np.linspace(0,2*np.pi, cpt_number, endpoint=False)]
                        if np.allclose(v1,v2): #起点和终点相同，完整圆
                            line_vertex_list = [start_idx] + list(range(pts_number,pts_number+cpt_number-1))#线段内点号重新定义
                            insert_points = whole_circle_points[1:]
                        else:
                            previous_angle = ct.get_angle_between_two_vectors(start_vector,v1,normxyz) #起始点距离圆起点夹角
                            previous_seg = int(0.5*previous_angle*cpt_number/np.pi) #前面有多少点需要跳过                            
                            end_angle = ct.get_angle_between_two_vectors(start_vector,v2,normxyz) # 圆初始点到圆弧终点夹角
                            if end_angle == 0: #终点和圆起点重合时
                                end_angle = np.pi*2
                            end_seg = int(0.5*end_angle*cpt_number/np.pi) #圆弧到哪个点结束  
                            if previous_seg < len(whole_circle_points)-1: #特例有不在whole circle 取点的
                                if np.allclose(whole_circle_points[previous_seg+1],circle_start,rtol=1e-9): #防止起点重复
                                    previous_seg+=1
                                #print(whole_circle_points[previous_seg+1],circle_start)
                            if end_seg < len(whole_circle_points): #如果圆弧终点不是圆终点，圆终点即为起点，不在 list 中
                                if np.allclose(whole_circle_points[end_seg],circle_end,rtol=1e-9): #如果与圆弧终点重合,防止终点重复
                                    end_seg-=1                                                                     
                            insert_points = whole_circle_points[previous_seg+1:end_seg+1] #插入点的坐标集合
                            line_vertex_list = [start_idx] + list(range(pts_number,pts_number+len(insert_points))) + [end_idx] #线段内点号重新定义
                        #保存
                        line_vertex_id_list.append(line_vertex_list);pts_xyz_list.extend(insert_points)
                        edge_curve.simulation_points = line_vertex_list
                    else: #plane
                        line_vertex_id_list.append([start_idx,end_idx])
                        edge_curve.simulation_points = [start_idx,end_idx]
                else: #line 已存在line_key_list
                    edge_curve.simulation_points = line_vertex_id_list[line_key_list.index(edge_curve.key)]

    model.line_list = line_vertex_id_list
    return line_key_list,line_type_list,line_vertex_id_list,pts_xyz_list

def remove_extra_vertex_in_facet(half_facet,coo_array):
    angles_in_facet = check_orthogonal(half_facet,coo_array)
    if angles_in_facet == [False, False, True, True]:#直边在下
        if len(half_facet[0]) > len(half_facet[1]):
            correct_facet = half_facet[0][:-1] + half_facet[1]
        else:
            correct_facet = half_facet[0] + half_facet[1][1:]
    elif angles_in_facet == [True, True, False, False]:#直边在上
        if len(half_facet[0]) > len(half_facet[1]):
            correct_facet = half_facet[0][1:] + half_facet[1]
        else:
            correct_facet = half_facet[0] + half_facet[1][:-1]
    else:
        raise NameError('检查组合面%s,angle%s'%(half_facet,angles_in_facet))  
    return correct_facet


def pair_cylindrical_lines_into_facets(cylindrical_lines,coo_array,whole_circle):       
    cylindrical_facets = []
    half_facet = None
    if whole_circle:
        cylindrical_lines_pairs = ct.tee(cylindrical_lines)
    else:
        cylindrical_lines_pairs = ct.tee(cylindrical_lines)[:-1]#如果不是完整圆，无需头尾相接
    
    # print(cylindrical_lines)
    # print('----')
    # print (cylindrical_lines_pairs)
    for pair_idx,line_pair in enumerate(cylindrical_lines_pairs): #把line 两两组合           
        cline_uniuqe = ct.list_unique_without_changing_order(line_pair[0]) #去除重复点
        next_cline_unique = ct.list_unique_without_changing_order(line_pair[1])#去除重复点
        facet = [cline_uniuqe,next_cline_unique[::-1]] #两线尾尾相连 
        angles_in_facet = check_orthogonal(facet,coo_array)
        if all(angles_in_facet):# 所有边依次垂直
            if not half_facet: #如果half_facet 存在,说明half_facet有冗余点
                cylindrical_facets.append((facet[0]+facet[1]))
            else:
                corrected_facet = remove_extra_vertex_in_facet(half_facet,coo_array)
                cylindrical_facets.append(corrected_facet)
                cylindrical_facets.append((facet[0]+facet[1]))
                half_facet = None 
        else:           
            if not half_facet: #如果half_facet 不存在
                if angles_in_facet.count(False) == 4:
                    facet = [cline_uniuqe,next_cline_unique] #弧线交接线z 字形，不需要倒置line2    
                #print(facet,angles_in_facet)
                half_facet = facet
            else:  
                if set(half_facet[1]) == set(facet[0]):#两面可以相连
                    facet = [half_facet[0]+half_facet[1],facet[1]]
                    angles_in_facet = check_orthogonal(facet,coo_array)
                    if all(angles_in_facet):# 所有边依次垂直
                        cylindrical_facets.append((facet[0]+facet[1]))   
                    else:
                        print(angles_in_facet)
                        raise NameError('检查圆弧结合线%s和%s,两面组合依然不合格'%(half_facet,facet))
                    half_facet = None      
                else:
                    raise NameError('检查圆弧结合线%s和%s'%(half_facet,facet))  
    if half_facet: #如果half_facet存在,说明最后一个面没处理好
        corrected_facet = remove_extra_vertex_in_facet(half_facet,coo_array)
        cylindrical_facets.append(corrected_facet) 
    return cylindrical_facets

def split_cylindrical_surface_into_lines(face_radius,face_center_xyz,face_normal_xyz,face_reference_direction,line_vertex_id_list,circle_edge,coo_array):
    project_length = [];arc_point_pool = []
    for arc in circle_edge: #获取圆弧在圆柱面法向上投影顺序
        arc_pts = line_vertex_id_list[arc][:]
        arc_point_pool.append(arc_pts) #save pts idx for pop
        arc_pts_coo = coo_array[arc_pts]
        project_length.append(np.mean(np.dot(arc_pts_coo,face_normal_xyz)))   
        # print(arc_pts)
        # print(arc_pts_coo)
        # print('*******')

    #确定从那个圆弧开始找
    # start_point_xyz =  face_center_xyz + face_reference_direction*face_radius
    # distances_from_start_vector = []
    # 
    # for aidx,arc in enumerate(circle_edge):
    #     if project_length[aidx] == np.amin(project_length):#底面圆，包含起始点
    #         distance = np.linalg.norm(coo_array[arc_point_pool[aidx][0]] - start_point_xyz)
    #     else:             
    #         distance = np.linalg.norm(np.cross(coo_array[arc_point_pool[aidx][0]] - \
    #         start_point_xyz,(np.dot(coo_array[arc_point_pool[aidx][0]],np.diag(face_normal_xyz)) - \
    #         np.dot(start_point_xyz,np.diag(face_normal_xyz)))))/ \
    #         np.linalg.norm(coo_array[arc_point_pool[aidx][0]] - start_point_xyz)#下一平面内圆弧点与法向垂直距离    
    #     distances_from_start_vector.append(distance) 

    distances_from_start_vector = []    
    start_vector_xyz = face_reference_direction*face_radius
    for aidx,arc in enumerate(circle_edge):
        circle_center_xyz = face_center_xyz + face_normal_xyz*(project_length[aidx]-np.dot(face_center_xyz,face_normal_xyz))
        circle_start_vector = coo_array[arc_point_pool[aidx][0]] - circle_center_xyz 
        angle = ct.get_angle_between_two_vectors(start_vector_xyz,circle_start_vector,face_normal_xyz)   
        distances_from_start_vector.append(angle) 
    
    search_order = [] #优先按水平排序，同等水平距离，按垂直距离
    while len(search_order)< len(circle_edge):
        for dist in np.unique(distances_from_start_vector):
            parallel_arcs =  np.nonzero(distances_from_start_vector==dist)[0] # 找到水平平行的圆弧哪个更靠近底部
            parallel_arcs_projection_order = np.argsort([project_length[arc] for arc in parallel_arcs])#圆弧按投影长度排序
            for parallel_idx in parallel_arcs_projection_order: #提取最底部的那个
                arc_idx = parallel_arcs[parallel_idx]
                if arc_idx not in search_order:
                    search_order.append(arc_idx) 
                    break
    #print (project_length,distances_from_start_vector,search_order)
    
    cylindrical_lines = []#把圆柱面圆弧点沿法向连接成切割线
    for aidx,arc in enumerate(search_order):#沿法向顺序挨个找直线点
        current_target_arc = None
        for pidx in arc_point_pool[arc][:]:
            split_line = [pidx]
            pcoo = coo_array[pidx]
            #print ('base',pidx,pcoo)
            for up in range(aidx+1,len(search_order)): #loop over next arcs
                if project_length[arc] != project_length[search_order[up]]: #非同平面                             
                    # angles = [np.dot(arc_vector - pcoo,face_normal_xyz)/(np.linalg.norm(arc_vector - pcoo) * np.linalg.norm(face_normal_xyz))
                    # for arc_vector in coo_array[arc_point_pool[search_order[up]]]] #下一平面内圆弧点与法向夹角
                    distances = [np.linalg.norm(np.cross(arc_pcoo - pcoo,(np.dot(arc_pcoo,np.diag(face_normal_xyz))-np.dot(pcoo,np.diag(face_normal_xyz)))))
                    /np.linalg.norm(arc_pcoo - pcoo) for arc_pcoo in coo_array[arc_point_pool[search_order[up]]]]#下一平面内圆弧点与法向垂直距离
                    if len(distances) > 0:#防止amin报错
                        #print(np.amin(distances))
                        if np.amin(distances) < 1e-9:
                            popp = arc_point_pool[search_order[up]][np.argmin(distances)]
                            #print (popp,coo_array[popp])
                            split_line.append(arc_point_pool[search_order[up]].pop(np.argmin(distances))) #取出此圆弧点
                            if current_target_arc != search_order[up]\
                                and np.argmin(distances)==1 and (current_target_arc != None): #切换下一条边的时候检查两边结合线
                                # print(distances[0], arc_point_pool[search_order[up]][0],\
                                # coo_array[ arc_point_pool[search_order[up]][0]],arc_point_pool[current_target_arc])
                                
                                last_arc_tail = arc_point_pool[current_target_arc][-1]
                                next_arc_head = arc_point_pool[search_order[up]][0]
                                edge_between_arc = coo_array[next_arc_head] -coo_array[last_arc_tail] #结合线
                                #print(last_arc_tail,next_arc_head,edge_between_arc)
                                if last_arc_tail == next_arc_head: #两圆弧相接
                                    cylindrical_lines.append([last_arc_tail,next_arc_head])
                                    arc_point_pool[current_target_arc].pop(-1)
                                    arc_point_pool[search_order[up]].pop(0)
                                elif ct.isParallel(edge_between_arc,face_normal_xyz): #检查结合线是否和法线一致
                                    cylindrical_lines.append([last_arc_tail,next_arc_head])
                                    arc_point_pool[current_target_arc].pop(-1)
                                    arc_point_pool[search_order[up]].pop(0)
                                elif ct.isAntiParallel(edge_between_arc,face_normal_xyz):
                                    cylindrical_lines.append([next_arc_head,last_arc_tail])
                                    arc_point_pool[current_target_arc].pop(-1)
                                    arc_point_pool[search_order[up]].pop(0)
                                else:
                                    print (np.argmin(distances),[search_order[up]],edge_between_arc)
                                    print(last_arc_tail,next_arc_head)
                                    print ('检查圆弧%d与%d边界线,坐标%s'%(circle_edge[current_target_arc],
                                        circle_edge[search_order[up]],coo_array[next_arc_head]))
                            current_target_arc = search_order[up]                                    
            if len(split_line) > 2:
                split_line = check_split_line_order(split_line,coo_array,face_normal_xyz)
            elif len(split_line)<2:
                raise NameError("检查圆柱面的分割情况,在点%s %s"%(pidx,pcoo))
            cylindrical_lines.append(split_line)
            #print('-------------------------------')
    return cylindrical_lines

def is_same_circle(ec1,ec2):   
    c1 = ec1.geometry.curve_3d
    c2 = ec2.geometry.curve_3d
    r1 = c1.radius
    r2 = c2.radius
    if r1 == r2:
        norm1 = c1.normal;normxyz1 = np.array([norm1.x,norm1.y,norm1.z])
        norm2 = c2.normal;normxyz2 = np.array([norm2.x,norm2.y,norm2.z])
        if np.allclose(normxyz1,normxyz2):
            cent1 = c1.center ; centxyz1 = np.array([cent1.x,cent1.y,cent1.z])
            cent2 = c2.center ; centxyz2 = np.array([cent2.x,cent2.y,cent2.z])
            if np.allclose(centxyz1,centxyz2):
                return True
            else:
                return False
        else:
            return False       
    else:
        return False

def check_whole_circle(circle_list):
    """check if there is complete circle in the circle list"""
    
    #group arcs in the same circle
    ecidx_list = list(range(len(circle_list)))
    groups = []
    for ecidx,edge_curve in enumerate(circle_list):
        if ecidx not in [c for g in groups for c in g]:
            group = [ecidx];ecidx_list.pop(ecidx_list.index(ecidx))
            for other in ecidx_list[:]:
                if is_same_circle(edge_curve,circle_list[other]):
                    group.append(other)
                    ecidx_list.pop(ecidx_list.index(other))
            groups.append(group)
    #find whole circle
    for circles in groups:
        total_angle = 0
        for cidx in circles:
            edge_curve = circle_list[cidx]
            circle = edge_curve.geometry.curve_3d
            cent = circle.center ; centxyz = np.array([cent.x,cent.y,cent.z])
            startxyz = edge_curve.start
            endxyz = edge_curve.end
            norm = circle.normal
            normxyz = np.array([norm.x,norm.y,norm.z])
            circle_start = np.array([startxyz.x,startxyz.y,startxyz.z])     
            circle_end = np.array([endxyz.x,endxyz.y,endxyz.z])
            v1 = circle_start - centxyz
            v2 = circle_end - centxyz
            if np.allclose(v1,v2):
                angle = np.pi*2
            else:
                angle = ct.get_angle_between_two_vectors(v1,v2,normxyz)             
            total_angle+= angle
        if total_angle >= np.pi*2:
            return True           
    return False

def two_arc_into_facets(line_vertex_id_list,circle_edge,face_normal_xyz,coo_array,whole_circle,bound):
    cylindrical_facets = []
    circle1_pts = line_vertex_id_list[circle_edge[0]]
    circle2_pts = line_vertex_id_list[circle_edge[1]]                
    if len(circle1_pts) == len(circle2_pts):#假设2个圆弧点数相等                    
        v1 = coo_array[circle2_pts[0]] -  coo_array[circle1_pts[0]]                                       
        if ct.isParallel(v1,face_normal_xyz):#平行  #检查是不是起点相同,定点排列方向
            for vertex_id in range(len(circle1_pts)-1):
                facet = [circle1_pts[vertex_id],circle1_pts[vertex_id+1],
                        circle2_pts[vertex_id+1],circle2_pts[vertex_id]]
                cylindrical_facets.append(facet)
            if whole_circle:
                cylindrical_facets.append([circle1_pts[-1],circle1_pts[0],
                                circle2_pts[0],circle2_pts[-1]])
        elif ct.isAntiParallel(v1,face_normal_xyz):#反向  
            for vertex_id in range(len(circle1_pts)-1):
                facet = [circle1_pts[vertex_id],circle2_pts[vertex_id],
                        circle2_pts[vertex_id+1],circle1_pts[vertex_id+1]]
                cylindrical_facets.append(facet)
            if whole_circle:
                cylindrical_facets.append([circle1_pts[-1],circle2_pts[-1],
                                circle2_pts[0],circle1_pts[0]])               
        else:
            raise NameError("检查圆柱面的圆弧方向%s"%circle_edge)
    else:
        raise NameError("检查圆柱面的边界定义%s"%bound.edge_list) 
    return cylindrical_facets

def check_stock_face(face_center,norm,coo_array):
    pts_dists = np.dot((coo_array-face_center),norm) 
    fctype = ct.is_same_sign(np.sign(pts_dists))
    return fctype

      
if __name__ == "__main__": 
    import os
    import sys
    import geometry_builder as gb
    os.chdir(sys.path[0])
    number = 12
    path = 'STEP/pocket_pocket/pocket1_pocket2_%d.step' %number
    path = '3D/blind_hole&pocket_%d.step'%number
    data = read_STEP(path)
    model = closed_shell(data)
    coo_array,facets,hole_facets = model.get_facets()
    my_model = gb.solid_model(coo_array,facets,hole_facets,min_length=1.5)
    #[print(fc) for fc in facets]
    print ('总共%d 面，%d个点'%(len(facets),coo_array.shape[0]))
