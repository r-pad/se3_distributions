import numpy as np
from scipy.linalg import det
#import mayavi.mlab as mlab
from .quaternion import Quaternion
from .rot3 import Rot3

# original file in python_modules/js/geometry/rotations.py
def plotCosy(fig,R,t,scale,name='',col=None):
  pts = np.zeros((3,6)) 
  for i in range(0,3):
    pts[:,i*2]  = np.zeros(3)
    pts[:,i*2+1] = R[:,i]
  pts *= scale
  pts+= t[:,np.newaxis]
  if col is None:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=(1.0,0.0,0.0),tube_radius=0.05*scale)
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=(0.0,1.0,0.0),tube_radius=0.05*scale)
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=(0.0,0.0,1.0),tube_radius=0.05*scale)  
  else:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=col)            
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=col)            
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=col) 
  if name!='':
    mlab.text3d(pts[0,1],pts[1,1],pts[2,1],name,
      scale=.1*scale,color=(0,0,0),line_width=1.0,figure=fig)
    mlab.text3d(pts[0,3],pts[1,3],pts[2,3],name,
      scale=.1*scale,color=(0,0,0),line_width=1.0,figure=fig)
    mlab.text3d(pts[0,5],pts[1,5],pts[2,5],name,
      scale=.1*scale,color=(0,0,0),line_width=1.0,figure=fig)

def ToRightQuaternionProductMatrix(x):
  return np.array([
    [0., -x[0], -x[1], -x[2]],
    [x[0], 0., x[2], -x[1]],
    [x[1], -x[2], 0., x[0]],
    [x[2], x[1], -x[0], 0.]])

def ToLeftQuaternionProductMatrix(x):
  return np.array([
    [0., -x[0], -x[1], -x[2]],
    [x[0], 0., -x[2], x[1]],
    [x[1], x[2], 0., -x[0]],
    [x[2], -x[1], x[0], 0.]])

if __name__ == "__main__":
  q = Quaternion(1.,0.,0.,0.)
  print(q.inverse())
  print(q.toAxisAngle())
  q2 = Quaternion(1.,.01,0.,0.)
  q2.normalize()
  print(q2)
  print(q2.inverse())
  print(q2.inverse().dot(q2))
  print(q2.dot(q2.inverse()))
  print(q.dot(q2))
  print(q.angleTo(q2))
  input()

  q2 = Quaternion(1.,1.,0.,0.)
  q2.fromAxisAngle(np.pi/2.0,np.array([0.,1.,1.]))

  q3 = q.slerp(q2,0.5)
  print(q)
  print(q3)
  print(q2)  
  for t in np.linspace(0.,1.,100):
    qi=q.slerp(q2,t)
    print("--------------------------------------------------------")
    print(qi)
    print(np.sqrt((qi.q**2).sum()))
    print('det: {}'.format( det(qi.toRot().R)))
    print("--------------------------------------------------------")
    #print qi.toRot().R


  dq = Quaternion()                                                                                                               
  q0 = Quaternion()                                                                                                               
  qe = Quaternion()                                                                                                               
  qe.fromAxisAngle(180,np.array([0.,0.,1.]))                                                                                      
  for t in np.linspace(0.0,1.0,10):                                                                                              
    # show pc colored according to their MF assignment                                                                            
    dq = q0.slerp(qe,t)
    print('-----------')
    print(dq)
    print(np.sqrt((qi.q**2).sum()))
    print('det: {}'.format( det(dq.toRot().R)))
    print(dq.toRot().R)

    print(dq.toRot().logMap())


