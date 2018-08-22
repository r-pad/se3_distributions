import numpy as np

def norm(v):
  return np.sqrt((v**2).sum())
def normed(v):
  return v/norm(v)

def vee(W):
  A = 0.5*(W - W.T)
  w = np.array([A[2,1],A[0,2],A[1,0]])
  return w

def invVee(w): # also called skew sometimes
  W = np.zeros((3,3))
  W[2,1] = w[0]
  W[0,2] = w[1]
  W[1,0] = w[2]
  W -= W.T # fill other  half of matrix
  return W

class Rot3:
  def __init__(self,R):
    if not (R.shape[0]==3 and R.shape[1]==3):
      raise notImplementedError
    self.R = R
  def toRPY(self):
    yaw   = np.arctan2(self.R[2,1],self.R[2,2])
    pitch = np.arctan2(-self.R[2,0],
        np.sqrt(self.R[2,1]**2+self.R[2,2]**2))
    roll  = np.arctan2(self.R[1,0],self.R[0,0])
    return np.array([roll,pitch,yaw])
  def toQuat(self):
    q = Quaternion()
    q.fromRot3(self.R)
    return q
  def logMap(self):
    theta = np.arccos((np.trace(self.R) -1)/2.0)
    W = 1./(2.*np.sinc(theta))*(self.R - self.R.T)
    #print theta/(2.*np.sin(theta))
    #print (self.R - self.R.T)
    return vee(W)
  def expMap(self,ww):
    if ww.size == 3:
      w = ww
      W = invVee(w)
    elif ww.shape[0] == 3 and ww.shape[1] == 3:
      W = ww
      w = vee(W)
    else:
      raise ValueError
    theta = np.sqrt((w**2).sum())
    a = np.sinc(theta)
    b = (1.-np.cos(theta))/(theta**2)
    if not b==b:
      b = 0.
#    if theta < 1e-12:
#      self.R = np.eye(3)
#    else:
#      self.R = np.eye(3) + np.sin(theta)/theta * W + (1.0 - np.cos(theta))/theta**2 * W.dot(W)
    self.R = np.eye(3) + a*W + b*W.dot(W)
    return self.R
  def dot(self,Rb):
    return Rot3(self.R.dot(Rb.R))
  def __repr__(self):
    return "{}".format(self.R)
