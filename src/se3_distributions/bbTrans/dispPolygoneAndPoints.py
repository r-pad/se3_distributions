import numpy as np

polygoneStr = "0.309017 0.5 0.809017 -0.309017 0.5 0.809017 -0.309017 -0.5 0.809017 0.309017 -0.5 0.809017"
pointsStr = "-0.309017 -0.5 0.809017 0 0.525731 0.850651 0 1 0"

polygoneStr = "0.309017 0.5 0.809017 -0.309017 0.5 0.809017 -0.309017 -0.5 0.809017 0.309017 -0.5 0.809017"
pointsStr = "-0.309017 -0.5 0.809017 0 0.525731 0.850651 0 1 0"


polygoneStr = "0.309017 0.5 -0.809017 -0.309017 0.5 -0.809017 -0.309017 -0.5 -0.809017 0.309017 -0.5 -0.809017"
pointsStr = "1 0 0 0.356822 0 -0.934172 -0.309017 0.5 -0.809017"

polygoneStr = "-0.5 0.809017 0.309017 0.5 0.809017 0.309017 0.5 0.809017 -0.309017 -0.5  0.809017 -0.309017"
pointsStr = "0 1 0 0 1 0 -0.5 0.809017 0.309017"

polygoneStr = "0.94047 0.315551 0.126266 0.957491 -0.253967  0.136785 0.824856 0.128484 0.550548 0.956795   0.289136 -0.0306946"
pointsStr = "1 0 0 0.957491 -0.253967  0.136785 0.824856 0.128484 0.550548"

print(polygoneStr)

mus = np.reshape(np.fromstring(polygoneStr, dtype=np.float, sep=" "),\
    (4,3))
ps = np.reshape(np.fromstring(pointsStr, dtype=np.float, sep=" "),
    (3,3))

#import mayavi.mlab as mlab
from js.geometry.sphere import Sphere
figm = mlab.figure(bgcolor=(1,1,1))
S = Sphere(2)
S.plotFanzy(figm,1.)
mlab.points3d(mus[:,0], mus[:,1], mus[:,2], scale_factor=0.1,
    color=(1,0,0), opacity=0.5)

mlab.points3d(ps[0,0], ps[0,1], ps[0,2], scale_factor=0.15, 
    color=(0,1,0), opacity=0.5)

mlab.points3d(ps[1,0], ps[1,1], ps[1,2], scale_factor=0.15, 
    color=(0,0,1), opacity=0.5)
mlab.points3d(ps[2,0], ps[2,1], ps[2,2], scale_factor=0.15, 
    color=(0,1,1), opacity=0.5)
mlab.show()

