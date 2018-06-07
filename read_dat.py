import numpy as np


def read_dat(part_name):
    """ Read Catia .dat mesh file with xyz coordinates and triangular mesh vertex connectivity
    :param part_name: string with the specified part name
    :return coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
         tri_array contains vertex connection indices of the part in array of shape=(#triangles, 3)
    """
    
    #store part name in a temp file for next step usage
    record = open('temp/name.txt','w')
    record.write('%s'%part_name)
    record.close()   
    
    print ("Reading %s file" %part_name)
    coord = []
    tri = []
    catia_version = ""
    with open(part_name, 'r') as fileId:
        for line in fileId:
            ele = line.split()
            if ele[0] == 'GRID*':
                # Check catia file version if catia_version is empty
                if not catia_version:
                    if len(ele) == 5:
                        catia_version = "V5R20"
                    else:
                        catia_version = "V5R19"
                    print("Catia file version %s" % catia_version)

                x_coord = float(ele[2])
                y_coord = float(ele[3])
            elif ele[0] == '*':
                if catia_version == "V5R20":
                    z_coord = float(ele[2])
                else:  # catia_version == "V5R19"
                    z_coord = float(ele[1])

                coord.append([x_coord, y_coord, z_coord])
            elif ele[0] == 'CTRIA3':
                tri.append([int(ele[3]), int(ele[4]), int(ele[5])])

    coord_array = np.array(np.reshape(coord, (-1, 3)))
    tri_array = np.array(np.reshape(tri, (-1, 3))) - 1  # Convert to Python index format starting at 0

    fileId.close()
    np.savez_compressed('temp/coo', coord_array=coord_array,tri_array=tri_array)
    return coord_array, tri_array
