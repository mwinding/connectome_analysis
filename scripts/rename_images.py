import os
import re
import numpy as np

path = '/Volumes/GoogleDrive/My Drive/blender/animations/all_brain_neurons/neuropil/'
files = os.listdir(path)
files = sorted(files)

#regex = re.compile(r'\d+')
#ints = [int(regex.findall(x)[0]) for x in files]
#ints = list

ints = [x*2 for x in range(1, 1+len(files))]
names = [str(x).zfill(6) + '.png' for x in ints]

for i in range(0, len(files)):
    os.rename(os.path.join(path, files[i]), os.path.join(path, names[i]))


path = '/Volumes/GoogleDrive/My Drive/blender/animations/all_brain_neurons/neuropil/'
files = os.listdir(path)
files = sorted(files)

ints = [x*2 for x in range(1, 1+len(files))]
names = [str(x).zfill(4) + '.png' for x in ints]

for i in range(0, len(files)):
    os.rename(os.path.join(path, files[i]), os.path.join(path, names[i]))
