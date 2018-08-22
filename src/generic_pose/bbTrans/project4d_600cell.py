import numpy as np
from scipy.linalg import solve, eig
#import mayavi.mlab as mlab
from .project4d import Project4d
from .discretized4dSphere import *
from itertools import combinations, permutations

S3 = S3Grid(0)

q = normed(S3.vertices[S3.tetra[0][0], :] + S3.vertices[S3.tetra[0][1], :] +
  S3.vertices[S3.tetra[0][2], :] + S3.vertices[S3.tetra[0][3], :])

qp = q*1.4
q0 = q
n = normed(q0-qp)
print(qp)
print(q0)
print(n)
cam = Project4d(qp, q0, n)

vs = S3.vertices
# project
vs3d = np.zeros((vs.shape[0],3))
for i in range(vs.shape[0]):
  vs3d[i,:] = cam.Project(vs[i,:])
# plot
gold = (1.,215/255.,0)
silver = (192/255.,192/255.,192/255.)
scale = 0.03
figm = mlab.figure(bgcolor=(1,1,1))
mlab.points3d(vs3d[:,0],vs3d[:,1],vs3d[:,2], color=gold, scale_factor=scale*10.)
xs , ys, zs, edges = [], [], [], []
n_points = 0
for tetra in S3.tetra:
  for comb in combinations(list(range(4)),2):
    i, j = tetra[comb[0]], tetra[comb[1]]
    xs.append(vs3d[i,0])
    xs.append(vs3d[j,0])
    ys.append(vs3d[i,1])
    ys.append(vs3d[j,1])
    zs.append(vs3d[i,2])
    zs.append(vs3d[j,2])
    edges.append([n_points*2, n_points*2+1])
    n_points += 1
# Create the points
src = mlab.pipeline.scalar_scatter(xs, ys, zs)
# Connect them
src.mlab_source.dataset.lines = edges
# The stripper filter cleans up connected lines
lines = mlab.pipeline.stripper(src)
# Finally, display the set of lines
mlab.pipeline.surface(lines, color=silver, line_width=3., opacity=.4)
mlab.show(stop=True)
