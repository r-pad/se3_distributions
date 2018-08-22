import numpy as np
#import mayavi.mlab as mlab
import time

from js.geometry.icosphere import IcoSphere

class SphereHistogram:
  def __init__(self, sphereGrid, level = None):
    self.sphereGrid = sphereGrid
    if level is None:
      self.level = sphereGrid.GetNumLevels()
    else:
      self.level = level
    self.hist = np.zeros(self.sphereGrid.GetNumTrianglesAtLevel(level))

  def Compute(self, pts):
    for pt in pts:
      self.hist[self.sphereGrid.GetTrianglID(pt)[1]] += 1.
    print(self.hist)

  def Plot(self, level, figm):
    self.sphereGrid.Plot(self.level, figm)
    

run_tests = False
sphere = IcoSphere(2)
######################################### Intersection tests
if run_tests:
  p0 = np.array([0,0,1])
  p1 = np.array([1,0,1])
  p2 = np.array([0,1,1])
  t0 = time.clock()
  q = np.array([0.5, 0.5, -0.5])
  if sphere.Intersects(p0, p1, p2, q):
    print("Wrong intersection detected {}".format(q))
  q = np.array([0.5, 0.5, 1.])
  if not sphere.Intersects(p0, p1, p2, q):
    print("intersection not detected {}".format(q))
  q = np.array([0.5, 0., 1.])
  if not sphere.Intersects(p0, p1, p2, q):
    print("intersection not detected {}".format(q))
  q = np.array([0., 0.5, 1.])
  if not sphere.Intersects(p0, p1, p2, q):
    print("intersection not detected {}".format(q))
  q = np.array([1., 0.0, 0.])
  p0 = np.array([ 0.96193836,  0., -0.27326653])
  p1 = np.array([ 1.,  0.,  0.])
  p2 = np.array([ 0.95105652,  -0.26286556, -0.16245985])
  if not sphere.Intersects(p0, p1, p2, q):
    print("intersection not detected {}".format(q))
  p1 = np.array([ 0.96193836,  0., -0.27326653])
  p0 = np.array([ 1.,  0.,  0.])
  p2 = np.array([ 0.95105652,  -0.26286556, -0.16245985])
  if not sphere.Intersects(p0, p1, p2, q):
    print("intersection not detected {}".format(q))
  t1 = time.clock()
  print("tests tock {} ms".format((t1-t0)*1e3))

figm = mlab.figure(bgcolor=(1,1,1))
for lvl in range(3):
  sphere.Plot(lvl, figm)
  mlab.show(stop=True)

#q = np.array([0.5, 0.5, -0.5])
#q /= np.sqrt((q**2).sum())
#sphere.GetTrianglID(q)

q = np.array([[0.5, 0.5, 0.5],
  [0.5, 0.5, -0.5],
  [0.5, -0.5, -0.5],
  [0.5, -0.5, -0.5],
  [0.5, -0.5, -0.5],
  ])
sphereHist = SphereHistogram(sphere, 2)
sphereHist.Compute(q)

figm = mlab.figure()
sphereHist.Plot(2, figm)
mlab.show(stop=True)
