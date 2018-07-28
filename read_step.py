# -*- coding: utf-8 -*-
import numpy as np
import re
import commontool as ct


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
                    print ('这有个错！%s'%line)
    key_pattern = re.compile(r'#\d+')
    xyz_pattern = re.compile(r'\((\d+\.\d*,\d+\.\d*,\d+\.\d*)\)')
    
    pts_id_list = []
    pts_xyz_list = []
    face_list = []
    face_direction_list = []
    for key in data:
        value = data[key]
        if value[0] == 'CLOSED_SHELL':
            advance_face_keys = key_pattern.findall(value[1]) #get the faces key
            for fck in advance_face_keys: #loop over all faces
                face_value = data[fck]
                if face_value[0] == 'ADVANCED_FACE':
                    face_detail = re.findall(r',\((#\d+)\),(#\d+),',face_value[1])
                    face_boundary_key = face_detail[0][0] 
                    face_plane_key = face_detail[0][1]
                    if data[face_plane_key][0] == 'PLANE':
                        plane_def_key = key_pattern.findall(data[face_plane_key][1])[0]
                        plane_placement = [] #预建
                        if data[plane_def_key][0] == 'AXIS2_PLACEMENT_3D':  
                            plane_info_key =  key_pattern.findall(data[plane_def_key][1])
                            plane_point_key = plane_info_key[0]
                            plane_v1_key = plane_info_key[1]
                            plane_v2_key = plane_info_key[2]
                            if data[plane_point_key][0] == 'CARTESIAN_POINT':
                                plane_point_xyz = xyz_pattern.findall(data[plane_point_key][1])[0]
                                plane_placement.append(plane_point_xyz) #save
                            else:
                                print ('请检查此PLANE定义%s,找不到CARTESIAN_POINT'%data[plane_def_key]) 
                            if data[plane_v1_key][0]==data[plane_v1_key][0]== 'DIRECTION':
                                plane_v1 = xyz_pattern.findall(data[plane_v1_key][1])[0] 
                                plane_v2 = xyz_pattern.findall(data[plane_v2_key][1])[0]
                                plane_placement.append(plane_v1) #save
                                plane_placement.append(plane_v2)
                            else:
                                print ('请检查此PLANE定义%s，找不到DIRECTION'%data[plane_def_key])                       
                        else:
                            print ('请检查此PLANE定义%s，找不到AXIS2_PLACEMENT_3D'%data[plane_def_key])         
                        face_direction_list.append(plane_placement) #save
                    else:
                        print ('请检查此PLANE定义%s，找不到PLANE'%data[face_plane_key]) 
                    if data[face_boundary_key][0] == 'FACE_BOUND':
                        boundary_loop_key = re.findall(r',(#\d+),',data[face_boundary_key][1])[0]
                        if data[boundary_loop_key][0] == 'EDGE_LOOP':
                            boundary_loop_edge_keys = re.findall(r'(#\d+)',data[boundary_loop_key][1])        
                            face_boudary_edge_list = []
                            for edge_key in boundary_loop_edge_keys:
                                oriented_edge = data[edge_key]
                                if oriented_edge[0] == 'ORIENTED_EDGE':
                                    edge_curve_key = key_pattern.findall(oriented_edge[1])[0]
                                    edge_curve = data[edge_curve_key]
                                    if edge_curve[0] == 'EDGE_CURVE':
                                        edge_curve_info = key_pattern.findall(edge_curve[1])
                                        edge_start_end = []
                                        for info_key in edge_curve_info:
                                            if data[info_key][0] == 'VERTEX_POINT': # start and end point
                                                vertex_point_key = key_pattern.findall(data[info_key][1])[0]
                                                edge_start_end.append(vertex_point_key)
                                                if vertex_point_key not in pts_id_list:
                                                    pts_id_list.append(vertex_point_key)
                                                    if data[vertex_point_key][0] == 'CARTESIAN_POINT':
                                                        point_xyz = xyz_pattern.findall(data[vertex_point_key][1])[0]
                                                        pts_xyz_list.append(point_xyz)
                                                        if len(point_xyz) == 0:
                                                            print ('请检查此CARTESIAN_POINT定义%s'%data[vertex_point_key])
                                                    else:
                                                        print ('请检查此VERTEX_POINT定义%s'%data[info_key]) 
                                            elif data[info_key][0] == 'SURFACE_CURVE':
                                                line_key = re.findall(r',(#\d+),\(.+\),',data[info_key][1])[0]
                                                pcurve_keys = re.findall(r',#\d+,\((.+)\),',data[info_key][1])
                                                pcurve_keys = key_pattern.findall(pcurve_keys[0])                            
                                                if data[line_key][0] == 'LINE':
                                                    line_info_keys = key_pattern.findall(data[line_key][1])
                                                    line_point_key = line_info_keys[0] 
                                                    line_vector_key = line_info_keys[1]
                                                    #未存
                                                else:
                                                    print ('请检查此line定义%s'%data[line_key])                               
                                                for pc_key in pcurve_keys:
                                                    if data[pc_key][0] == 'PCURVE':  
                                                        pcurve_info_keys = key_pattern.findall(data[pc_key][1])
                                                        plane_key = pcurve_info_keys[0]
                                                        definition_representation_key = pcurve_info_keys[1]#未
                                                        if data[plane_key][0] == 'PLANE':
                                                            plane_def_key = key_pattern.findall(data[plane_key][1])[0]
                                                            if data[plane_def_key][0] == 'AXIS2_PLACEMENT_3D':  
                                                                plane_info_key =  key_pattern.findall(data[plane_def_key][1])
                                                                plane_point_key = plane_info_key[0]
                                                                plane_v1_key = plane_info_key[1]
                                                                plane_v2_key = plane_info_key[2]
                                                                if data[plane_point_key][0] == 'CARTESIAN_POINT':
                                                                    plane_point_xyz = xyz_pattern.findall(data[plane_point_key][1])[0]
                                                                else:
                                                                    print ('请检查此PLANE定义%s'%data[plane_def_key]) 
                                                                if data[plane_v1_key][0]==data[plane_v1_key][0]== 'DIRECTION':
                                                                    plane_v1 = xyz_pattern.findall(data[plane_v1_key][1])[0] 
                                                                    plane_v2 = xyz_pattern.findall(data[plane_v2_key][1])[0]
                                                                    #未存
                                                                else:
                                                                    print ('请检查此PLANE定义%s'%data[plane_def_key])                       
                                                            else:
                                                                print ('请检查此PLANE定义%s'%data[plane_def_key]) 
                                                
                                                        else:
                                                            print ('请检查此PLANE定义%s'%data[plane_key]) 
                                                        if data[definition_representation_key][0] == 'DEFINITIONAL_REPRESENTATION':
                                                            line_key = re.findall(r',\((#\d+)\),#\d+',data[definition_representation_key][1])[0]
                                                            geo_rep_context_key = re.findall(r',\(#\d+\),(#\d+)',data[definition_representation_key][1])[0]# 未处理                        
                                                            if data[line_key][0] == 'LINE':
                                                                line_info_keys = key_pattern.findall(data[line_key][1])
                                                                line_point_key = line_info_keys[0] 
                                                                line_vector_key = line_info_keys[1]
                                                                # 未存
                                                            elif data[line_key][0] == 'B_SPLINE_CURVE_WITH_KNOTS':
                                                                pass
                                                            
                                                            else:
                                                                print ('请检查此line定义%s'%data[line_key]) 
                                                        else:
                                                            print ('请检查此DEFINITIONAL_REPRESENTATION定义%s'%data[definition_representation_key]) 
                                            
                                                    else:
                                                        print ('请检查此Pcurve定义%s'%data[pc_key]) 
                                            else:
                                                print ('请检查此edge_curve定义%s'%data[info_key])
                                    else:
                                        print ('请检查此edge_curve定义%s'%edge_curve)
                                    face_boudary_edge_list.append(edge_start_end)
                                else:
                                    print ('请检查此边定义%s'%oriented_edge)
                        else:
                            print ('请检查此loop定义%s'%data[boundary_loop_key] )
                    else:
                        print ('请检查此面边界定义%s'%data[face_boundary_key])
                    face_list.append(face_boudary_edge_list)
                else:
                    print ('请检查此面定义%s'%face_value)

    coo_array = np.array([[float(x)for x in re.split(r',',xyz)] for xyz in pts_xyz_list]) #convert string to numbers
    
    #convert #id into index in coo_arry
    edge_loop_list = [[[pts_id_list.index(vid) for vid in edge] for edge in fc] for fc in face_list]
    
    # 0 original point, 1 normal vector of plane, 2 reference axis ???
    face_placement_array =  [[[float(x)for x in re.split(r',',xyz)] for xyz in fc] for fc in face_direction_list]
    
    point_loop_list = [ct.edge_loop2point_loop(loop) for loop in edge_loop_list] 
    
    face_normals,new_point_loop_list = ct.get_normal_from_model(coo_array,point_loop_list)
    
    return coo_array,new_point_loop_list

if __name__ == "__main__": 
    import os
    import sys
    os.chdir(sys.path[0])
    path = '3D/slot_pocket_0.step' #test sample path
    coo,facets = read_STEP(path)