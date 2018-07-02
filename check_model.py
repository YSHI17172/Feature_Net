import numpy as np

data = np.load('input_mesh/through_hole/through_hole_143.npz')
model = data['model'][()]

print ('total feature %d'%len(model.features))
print ('total faces %d'%len(model.faces))
print (len(model.features[0].faces))
print (model.features[0].faces)