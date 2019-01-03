import os

feature = 'pocket_pocket'
path = 'STEP/%s/'%feature#test sample path
save_path = '/Volumes/ExFAT256/clusters/%s/'%feature

exsiting_files= os.listdir(save_path)
name_len = len(feature)+1
exsiting_numbers = [int(name[16:-4]) for name in exsiting_files if '_' in name]

to_do_number = [n for n in range(5000) if n not in exsiting_numbers]
print(to_do_number)

#a = [2076, 2094, 2124, 2134, 2296, 2480, 2580, 2747, 2869, 3043, 3161, 3183, 3259, 3301, 3316, 3433, 3525, 3549, 3578, 3700, 3770, 3863, 3921, 4003, 4035, 4100, 4175, 4272, 4406, 4433, 4532, 4576, 4716, 4767, 4781, 4835, 4908, 4983]

#print([x-4000 for x in a])