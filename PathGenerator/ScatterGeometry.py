import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('tkagg')

import sys
import json
import torch

file = sys.argv[1]

params = json.load(open("PathGenerator/Paths/"+file+".json","r"))

positions = torch.FloatTensor(params["positions"])

coords = torch.permute(positions,(0,2,1))


fig = plt.figure()

ax = fig.add_subplot(projection='3d')
for i,xyzs in enumerate(coords):
    if i == 0:
        label = "Start"
    elif i == len(coords)-1:
        label = "End"
    else:
        label = "Target "+str(i+1)
    ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker="o",label=label)


plt.xlim((-6,6))
plt.ylim((-6,6))
ax.set_zlim((-6,6))

plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")

plt.legend()
plt.show()