import numpy as np
from scipy.linalg import solve, eig
#import mayavi.mlab as mlab

def norm(a):
  return np.sqrt((a**2).sum())
def normed(a):
  return a/np.sqrt((a**2).sum())

class Project4d(object):
  def __init__(self, qp, q0, n):
    ''' Projection onto hyperplane at p0 with normal n'''
    self.q0 = q0
    self.qp = qp
    self.n = n

    # Gram Schmidt orthonormalization.
    self.U = np.eye(4)
    self.U[:,0] = normed(self.n)
    for i in range(1,4):
      for j in range(0,i):
        self.U[:,i] -= \
          ((self.U[:,i].dot(self.U[:,j]))/(self.U[:,j].dot(self.U[:,j])))*self.U[:,j]
    self.E = np.eye(4)
    for i in range(4):
      self.E[:,i] = normed(self.U[:,i])
    for i in range(4):
      for j in range(i+1,4):
        print(i,j, np.arccos(self.E[:,i].dot(self.E[:,j]))*180./np.pi)
#    print n
#    print self.E
  def Project(self, p):
    t = (p - self.qp).T.dot(self.n)/(self.q0 - self.qp).T.dot(self.n)
    x = t*p + (1.-t)*self.qp 
    return solve(self.E[:,1:].T.dot(self.E[:,1:]),
        self.E[:,1:].T.dot(self.q0 - x))

if __name__ == "__main__":
  # construct a tesseract
  vs = np.zeros((16,4))
  i=0
  for a in [-1.,1.]:
    for b in [-1.,1.]:
      for c in [-1.,1.]:
        for d in [-1.,1.]:
          vs[i,:] = [a,b,c,d]
          i+=1
  edges = []
  for i in range(vs.shape[0]):
    for j in range(vs.shape[0]):
      if not i == j and norm(vs[i,:]-vs[j,:]) == 2.:
        edges.append([i,j])
  # select projection parameters -> center the projection on one of the
  # sides of the tesseract => Schlegel diagram!
  qp = np.array([-2.,0.,0.,0.])
  q0 = np.array([-1.,0.,0.,0.])
  n = normed(q0-qp)
  cam = Project4d(qp, q0, n)
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
  for edge in edges:
    i, j = edge
    mlab.plot3d([vs3d[i,0],vs3d[j,0]],[vs3d[i,1],vs3d[j,1]],[vs3d[i,2],vs3d[j,2]],
        tube_radius=scale, color=silver)
  mlab.show(stop=True)
