import numpy as np
from .rot3 import Rot3

def normed(a):
  return a/np.sqrt((a**2).sum())

class Quaternion:
  def __init__(self,w=1.0,x=0.0,y=0.0,z=0.0,vec=None):
    self.q =np.array([w,x,y,z])
    if not vec is None and vec.size == 3:
      self.q =np.array([w,vec[0],vec[1],vec[2]])
    if not vec is None and vec.size == 4:
      self.q = vec.copy()
    self.flipToUpperHalfSphere()
  def flipToUpperHalfSphere(self):
    if self.q[0] < 0:
      self.q = -self.q # flip onto upper half sphere
  def setToRandom(self):
    self.q = normed(np.random.randn(4))
    self.flipToUpperHalfSphere()
  def fromRot3(self,R_):
    # https://www.cs.cmu.edu/afs/cs/academic/class/16741-s07/www/lecture7.pdf
    if isinstance(R_,Rot3):
      R = R_.R
    else:
      R = R_
    qs = np.zeros(4)
    qs[0] = 1.0+R[0,0]+R[1,1]+R[2,2]
    qs[1] = 1.0+R[0,0]-R[1,1]-R[2,2]
    qs[2] = 1.0-R[0,0]+R[1,1]-R[2,2]
    qs[3] = 1.0-R[0,0]-R[1,1]+R[2,2]
    iMax = np.argmax(qs)
    self.q[iMax] = 0.5*np.sqrt(qs[iMax])
    if iMax ==0:
      self.q[1] = 0.25*(R[2,1]-R[1,2])/self.q[0]
      self.q[2] = 0.25*(R[0,2]-R[2,0])/self.q[0]
      self.q[3] = 0.25*(R[1,0]-R[0,1])/self.q[0]
    elif iMax==1:
      self.q[0] = 0.25*(R[2,1]-R[1,2])/self.q[1]
      self.q[2] = 0.25*(R[0,1]+R[1,0])/self.q[1]
      self.q[3] = 0.25*(R[2,0]+R[0,2])/self.q[1]
    elif iMax==2:
      self.q[0] = 0.25*(R[0,2]-R[2,0])/self.q[2]
      self.q[1] = 0.25*(R[0,1]+R[1,0])/self.q[2]
      self.q[3] = 0.25*(R[2,1]+R[1,2])/self.q[2]
    elif iMax==3:
      self.q[0] = 0.25*(R[1,0]-R[0,1])/self.q[3]
      self.q[1] = 0.25*(R[0,2]+R[2,0])/self.q[3]
      self.q[2] = 0.25*(R[2,1]+R[1,2])/self.q[3]
    self.flipToUpperHalfSphere()
  def inverse(self):
    q = self.normalized().q
    return Quaternion(w=q[0],vec=-q[1:])
  def dot(self,p_):
    # horns paper
    r = self.q
    q = p_.q
    return Quaternion(
        w = r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
        x = r[0]*q[1]+r[1]*q[0]+r[2]*q[3]-r[3]*q[2],
        y = r[0]*q[2]-r[1]*q[3]+r[2]*q[0]+r[3]*q[1],
        z = r[0]*q[3]+r[1]*q[2]-r[2]*q[1]+r[3]*q[0])
#        w = q[0]*p[0]-q[1]*p[1]-q[2]*p[2]-q[3]*p[3],
#        x = q[0]*p[1]+q[1]*p[0]+q[3]*p[2]-q[2]*p[3],
#        y = q[0]*p[2]-q[3]*p[1]+q[2]*p[0]+q[1]*p[3],
#        z = q[0]*p[3]+q[2]*p[1]-q[1]*p[2]+q[3]*p[0])
  def angleTo(self,q2):
#    theta,_ = (self.dot(q2.inverse()).normalized()).toAxisAngle()
    print(self.q)
    print(q2.q)
    print(q2.inverse().q)
    dq = self.dot(q2.inverse()).normalized()
    theta = 2.*np.arctan2(np.sqrt((dq.q[1:]**2).sum()), dq.q[0])
    return theta
  def toAxisAngle(self):
    self.normalize()
#    theta = 2.0*np.arccos(self.q[0])
    theta = 2.*np.arctan2(np.sqrt((self.q[1:]**2).sum()), self.q[0])
    sinThetaHalf = np.sqrt(1.-self.q[0]**2)
    if theta < 1e-5:
      axis = np.array([0,0,1])
    else:
      axis = self.q[1::]/sinThetaHalf
    return theta,axis
  def toRPY(self):
    w,x,y,z = self.q[0],self.q[1],self.q[2],self.q[3]
    roll = np.arctan2(2*y*w-2*x*z,1.-2*y*y - 2*z*z)
    pitch = np.arctan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z)
    yaw = np.arcsin(2*x*y + 2*z*w)
    return np.array([roll,pitch,yaw])
  def fromAxisAngle(self,theta,axis):
    self.q[0] = np.cos(theta*0.5)
    self.q[1::] = axis*np.sin(theta*0.5)
  def normalize(self):
    self.q /= np.sqrt((self.q**2).sum())
  def normalized(self):
    norm = np.sqrt((self.q**2).sum())
    return Quaternion(self.q[0]/norm,self.q[1]/norm,self.q[2]/norm,self.q[3]/norm)
  def toAngularRate(self,dt):
    ax,theta = self.toAxisAngle()
    return ax*theta/dt
  def slerp(self,q2,t):
    # http://www.arcsynthesis.org/gltut/Positioning/Tut08%20Interpolation.html
    a = Quaternion()
    dot = self.q.dot(q2.q)
    if dot > 0.9995:
      #If the inputs are too close for comfort, 
      # linearly interpolate and normalize the result.
      a.q = self.q + t*(q2.q - self.q);
      a.normalize()
      return a;
    dot = min(max(dot,-1.),1.)
    theta_0 = np.arccos(dot);  # theta_0 = angle between input vectors
    theta = theta_0*t;  # theta = angle between v0 and result 
    a.q = q2.q - self.q*dot
    a.normalize()
    a.q = self.q*np.cos(theta) + a.q*np.sin(theta)
    return a
  def toRot(self): # LHCS??
    # this is from wikipedia
    a,b,c,d = self.q[0],self.q[1],self.q[2],self.q[3]
    R = np.array([[a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                  [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                  [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])
    return Rot3(R)
  def toRotOther(self): # RHCS  ??
    # http://osdir.com/ml/games.devel.algorithms/2002-11/msg00318.html
    a,b,c,d = self.q[0],self.q[1],self.q[2],self.q[3]
#    R = np.array([[a**2+b**2-c**2-d**2, 2*b*c+2*a*d, 2*b*d+2*a*c],
#                  [2*b*c-2*a*d, a**2-b**2+c**2-d**2, 2*c*d+2*a*b],
#                  [2*b*d+2*a*c, 2*c*d-2*a*b, a**2-b**2-c**2+d**2]])
    R = np.array([[1. -2.*c**2-2.*d**2, 2*b*c+2*a*d, 2*b*d-2*a*c],
                  [2*b*c-2*a*d, 1. -2.*b**2-2.*d**2, 2*c*d+2*a*b],
                  [2*b*d+2*a*c, 2*c*d-2*a*b, 1.-2.*b**2-2.*c**2]])
    return Rot3(R)
#  def plot(self, figm, t=None, scale=1., name=''):
#    if t is None:
#      t = np.zeros(3)
#    plotCosy(figm,self.toRot().R,t,scale,name)
  def rotate(self,v):
#    vn = normed(v)
#    vq = Quaternion(w=0.,vec=v)
#    vq = self.dot(vq.dot(self.inverse()))
#    return vq.q[1:] #*norm(v)
    # from Eigen
    # http://eigen.tuxfamily.org/dox/Quaternion_8h_source.html
    uv = np.cross(self.q[1::],v)
    uv += uv
    return v + self.q[0] * uv + np.cross(self.q[1::], uv)

  def __repr__(self):
#    return "{}".format(self.q)
    return "w|x|y|z: {}\t{}\t{}\t{}".format(self.q[0],self.q[1],self.q[2],self.q[3])


if __name__=="__main__":
  q0 = Quaternion()
  q1 = Quaternion()
  q0.fromAxisAngle(90./180.*np.pi, np.array([0,1,0]))
  q1.fromAxisAngle(90./180.*np.pi, np.array([0,1,0]))

  print(q0.angleTo(q1)*180./np.pi)

  v = np.array([ 0.884004, -0.371412, 0.115123, 0.259491])
  qa = Quaternion(vec=v) 
  qb = Quaternion(vec=-v) 
  print(qa.angleTo(qb)*180/np.pi)
  print(qa.toAxisAngle())
  print(qb.toAxisAngle())
  print(qa.dot(qb.inverse()).q)
  print(qa.dot(qb.inverse()).toAxisAngle())
