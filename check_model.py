import numpy as np

data = np.load('input_mesh/slot&pocket/slot&pocket_0.npz')
model = data['model'][()]

print ('total feature %d'%len(model.features))
print ('total faces %d'%len(model.faces))
for i,feature in enumerate(model.features):
    print (len(model.features[i].faces))
    print (model.features[i].faces)
    print ('')

for face in model.faces:
    print(face.ID)
    print('face type',face.type)
    # for loop in face.loops:
    #     print(loop.line_list)
    #     print('type',loop.type)
    print('')