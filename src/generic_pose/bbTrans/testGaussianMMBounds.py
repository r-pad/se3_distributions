import numpy as np
from scipy.linalg import det, solve, inv


class Box(object):
  def __init__(self, ld, ru):
    self.ld = ld
    self.ru = ru
    self.lu = np.array([self.ld[0],self.ru[1]])
    self.rd = np.array([self.ru[0],self.ld[1]])
  def Inside(self, x):
    return (self.ld <= x).all() and (x <= self.ru).all()
  def GetEdge(self, i):
    if i == 0:
      return self.ld, -self.ld + self.lu
    elif i == 1:
      return self.lu, -self.lu + self.ru
    elif i == 2:
      return self.ru, -self.ru + self.rd
    elif i == 3:
      return self.rd, -self.rd + self.ld
  def GetMiddle(self):
    return (self.ld+self.ru)*0.5

class Gaussian(object):
  def __init__(self, mu, Sigma, pi):
    self.pi = pi
    self.mu = mu
    self.Sigma = Sigma
    self.D = mu.size
  def pdf(self, x):
    return (2.*np.pi)**(-self.D*0.5) / np.sqrt(det(self.Sigma)) \
    * np.exp(-0.5*(x-self.mu).T.dot(solve(self.Sigma, x-self.mu)))
  def logPdf(self, x):
    return np.log(2.*np.pi)*(-self.D*0.5) - 0.5*np.log(det(self.Sigma))\
  -0.5*(x-self.mu).T.dot(solve(self.Sigma, x-self.mu))
  def GetZ(self):
    return (2.*np.pi)**(-self.D*0.5) / np.sqrt(det(self.Sigma))

def ComputeGmmForT(gmmA, gmmB, R):
  gmmT = []
  for gA in gmmA:
    for gB in gmmB:
      gmmT.append(Gaussian(-gA.mu + R.dot(gB.mu), gA.Sigma + \
        R.dot(gB.Sigma.dot(R.T)), gA.pi*gB.pi))
      print(gmmT[-1].mu.ravel(), gmmT[-1].pi)
#  Gamma_global = np.zeros(len(gmmT))
  Gamma_jk = np.zeros((len(gmmT),len(gmmT)))
  for k, gTk in enumerate(gmmT):
#    tU = FindMaxTinBox(inv(gTk.Sigma), solve(gTk.Sigma, gTk.mu),
#      box_max)
    if False: 
      print('--', t.ravel())
      plt.figure()
      for i in range(4):
        a,_ = box.GetEdge(i)
        plt.plot(a[0],a[1],'ro')
        print(a.ravel())
      plt.plot(tU[0],tU[1],'bx')
      plt.plot(gTk.mu[0],gTk.mu[1],'gx')
      plt.xlim([-0.1,2.1])
      plt.ylim([-0.1,2.1])
      plt.show()
    for j, gTj in enumerate(gmmT):
      if not k == j:
        Gamma_jk[j,k] = gTj.pi*gTj.GetZ()/(gTk.pi*gTk.GetZ())
#        Gamma_global[k] += Gamma_jk[j,k] * \
#          np.exp(0.5*(tU-gTk.mu).T.dot(solve(gTk.Sigma, tU-gTk.mu)))
  A = []
  b = []
  for k,gT in enumerate(gmmT):
    A.append(inv(gT.Sigma))
    b.append(solve(gT.Sigma, gT.mu))
#    A += Gamma_global[k] * inv(gT.Sigma)
#    b += Gamma_global[k] * solve(gT.Sigma, gT.mu)
  return gmmT, A, b,  Gamma_jk

def LowerBound(gmmT, box):
  lb = 0;
  for gT in gmmT:
    lb += gT.pi * gT.pdf(box.GetMiddle())
  lbs = np.zeros(len(gmmT))
  for k,gT in enumerate(gmmT):
    lbs[k] = np.log(gT.pi) + gT.logPdf(box.GetMiddle())
  print("LB", lb, np.exp(lbs-lbs.max()).sum()*np.exp(lbs.max()))
  if False: 
    print('--', t.ravel())
    plt.figure()
    for i in range(4):
      a,_ = box.GetEdge(i)
      plt.plot(a[0],a[1],'ro')
    plt.plot(box.GetMiddle()[0],box.GetMiddle()[1],'bx')
    plt.xlim([-0.1,2.1])
    plt.ylim([-0.1,2.1])
    plt.show()
  return lb

def JensenLowerBound(gmmT, box):
  lbs = np.ones(len(gmmT));
  for i,gT in enumerate(gmmT):
    lbs[i] = gT.pi * gT.logPdf(box.GetMiddle())
  return (lbs.sum())

def UpperBound(gmmT, box):
  ubs = np.ones(len(gmmT));
  for i,gT in enumerate(gmmT):
    t = FindMinTinBox(inv(gT.Sigma), solve(gT.Sigma, gT.mu), box)
    ubs[i] = gT.pi * gT.pdf(t) 
#    if box.Inside(gT.mu):
#      ubs[i] = gT.pi * gT.pdf(gT.mu) # can make this faster
#    else:
#      vals = np.zeros(4)
#      for i in range(4):
#        a,d = box.GetEdge(i)
#        # This is finding the max!
##        beta = 2.*(a+gT.mu).T.dot(solve(gT.Sigma, d))
##        alpha = d.T.dot(solve(gT.Sigma, d))
##        tau = -2.*beta/alpha
##        print alpha, tau
##        tau = min(1.,max(0.,tau))
##        vals[i] = gT.pdf(a+tau*d)
#        vals[i] = gT.pdf(a)
#      ubs[i] = np.min(vals)
    if not box.Inside(t):
      print('--', t.ravel())
      plt.figure()
      for i in range(4):
        a,_ = box.GetEdge(i)
        plt.plot(a[0],a[1],'ro')
        print(a.ravel())
      plt.plot(t[0],t[1],'bx')
      plt.plot(gT.mu[0],gT.mu[1],'kx')
      plt.show()
  return ubs.sum()

def FindMinTinBox(A,b,box,All=False):
  ts = []
  vals =[]
  t = solve(A, b)
  #print A,b,t
  if not box.Inside(t):
    # check the sides for the min
    for i in range(4):
      a, d = box.GetEdge(i)
      alpha = (d.T.dot(b) - d.T.dot(A).dot(a))/(d.T.dot(A).dot(d)) 
      if 0. <= alpha and alpha <= 1.:
        t = a+alpha*d
        ts.append(t)
        vals.append( t.T.dot(A).dot(t) -2.*t.T.dot(b))
#        print "box edge: ", a,d
#        print (d.T.dot(b)), "over", (d.T.dot(A).dot(d)) 
#        print b.ravel(), d.ravel()
#        print alpha, t, vals[-1]
    for i in range(4):
      t,_ = box.GetEdge(i)
      ts.append(t)
      vals.append( t.T.dot(A).dot(t) -2.*t.T.dot(b))
    i_min = np.argmin(vals)
    t = ts[i_min]
#    print vals
#    print ts
  if not box.Inside(t):
    print("WARNING sth is wrong here - computed t outside of given box.")
    print(t)
  if All:
    return t, ts, vals
  return t

def FindMaxTinBox(A,b,box):
  # check the corners for the max
  vals = []
  ts = []
  for i in range(4):
    t,_ = box.GetEdge(i)
    ts.append(t)
    vals.append(t.T.dot(A).dot(t) -2.*t.T.dot(b))
  i_max = np.argmax(vals)
  t = ts[i_max]
  return ts[i_max]

def UpperBound2(gmmT, A, b, Gamma_jk, box):
  ''' log inequality '''
  Gamma = np.ones(len(gmmT))
  for k,gTk in enumerate(gmmT):
    tU = FindMaxTinBox(A[k], b[k], box)
    logU = 0.5*(tU-gTk.mu).T.dot(solve(gTk.Sigma, tU-gTk.mu))
#    tL = FindMinTinBox(A[k], b[k], box)
    for j,gTj in enumerate(gmmT):
      tL = FindMinTinBox(A[k], b[k], box)
      logL = 0.5*(tL-gTj.mu).T.dot(solve(gTj.Sigma, tL-gTj.mu))
      print(k,"logs", logU, logL, np.exp(logU-logL), Gamma_jk[j,k])
      Gamma[k] += Gamma_jk[j,k] * np.exp(logU - logL)
  Gamma = 1./Gamma
  A_=np.zeros((2,2))
  b_=np.zeros((2,1))
  for k,gTk in enumerate(gmmT):
    A_ += Gamma[k] * A[k]
    b_ += Gamma[k] * b[k]
  print(A_, b_)
  t = FindMinTinBox(A_,b_,box)
  print("gamma",Gamma)
  ubs = np.zeros(len(gmmT)) 
  for k,gT in enumerate(gmmT):
    print("logpdf at ", t.ravel(), gT.logPdf(t), Gamma[k])
    ubs[k] = Gamma[k] * (np.log(gT.pi) + gT.logPdf(t))
  print(t, ubs)
  if False:
    plt.figure()
    for i in range(4):
      a,_ = box.GetEdge(i)
      plt.plot(a[0],a[1],'ro')
#      print a
    plt.plot(t[0],t[1],'bx')
    plt.xlim([0,2])
    plt.ylim([0,2])
    plt.show()
  return np.exp(ubs.sum()) * len(gmmT)

def UpperBoundConvexity(gmmT, box):
  ''' log inequality '''
  A_=np.zeros((2,2))
  b_=np.zeros((2,1))
  c_ = 0
  for k,gT in enumerate(gmmT):
    A, b = inv(gT.Sigma), solve(gT.Sigma, gT.mu)
    tU = FindMinTinBox(A, b, box)
    tL = FindMaxTinBox(A, b, box)
    L = -0.5*(tL-gT.mu).T.dot(solve(gT.Sigma, tL-gT.mu))
    U = -0.5*(tU-gT.mu).T.dot(solve(gT.Sigma, tU-gT.mu))
    g = (1.-np.exp(L-U))*np.exp(U)/(U-L)
    h = (np.exp(L-U)*U-L)*np.exp(U)/(U-L)
    print('L,U', L,U, g, h)
    D = gT.pi * (2.*np.pi)**(-1.) / np.sqrt(det(gT.Sigma))
    A_ -= 0.5*D*g*A
    b_ += D*g*b
    c_ += D*(h-0.5*g*gT.mu.T.dot(b))
#    print g, h, -0.5*D*g,D*g, D*(h-0.5*g*gT.mu.T.dot(b))
    if False:
      plt.figure()
      for i in range(4):
        a,_ = box.GetEdge(i)
        plt.plot(a[0],a[1],'ro', ms=6.)
      plt.plot(tU[0],tU[1],'bx', ms=20.)
      plt.plot(tL[0],tL[1],'gx', ms=20.)
      plt.plot(gT.mu[0],gT.mu[1],'bo', ms=20.)
      plt.xlim([-0.1,2.1])
      plt.ylim([-0.1,2.1])
      plt.show()
  # Not this one?
  t,ts,vals = FindMinTinBox(-A_,0.5*b_,box, True)
  ub1 = t.T.dot(A_.dot(t)) + b_.T.dot(t) + c_
  print('ub', ub1, t.T.dot(A_.dot(t)), b_.T.dot(t), c_)
  print(-0.25*b_.T.dot(solve(A_,b_)) + c_)
#  ubs = np.zeros(len(gmmT))
#  for k,gT in enumerate(gmmT):
#    A, b = inv(gT.Sigma), solve(gT.Sigma, gT.mu)
#    tL = FindMaxTinBox(A, b, box)
#    tU = FindMinTinBox(A, b, box)
#    L = -0.5*(tL-gT.mu).T.dot(solve(gT.Sigma, tL-gT.mu))
#    U = -0.5*(tU-gT.mu).T.dot(solve(gT.Sigma, tU-gT.mu))
##    print L,U
#    g = (1.-np.exp(L-U))*np.exp(U)/(U-L)
#    h = (np.exp(L-U)*U-L)*np.exp(U)/(U-L)
##    print g, h
#    D = gT.pi * (2.*np.pi)**(-1.5) / det(gT.Sigma)
##    print D*g
#    A_ = -0.5*D*g*A
#    b_ = D*g*b
#    c_ = D*(h-0.5*g*gT.mu.T.dot(b))
#    ubs[k] = t.T.dot(A_.dot(t)) + b_.T.dot(t) + c_
#  print ubs
#  print ub, ub1
  if False:
    plt.figure()
    for i in range(4):
      a,d = box.GetEdge(i)
      plt.plot(a[0],a[1],'ro')
      plt.plot([a[0], a[0]+d[0]],[a[1],a[1]+d[1]],'r-')
#      print a
    m = solve(A_,-0.5*b_)
    M = solve(-A_, 0.5*b_)
    print("m", m.ravel(), M.ravel())
    print(vals)
    plt.plot(t[0],t[1],'bx',ms=6)
    plt.plot(m[0],m[1],'bo',ms=11)
    plt.plot(M[0],M[1],'rx',ms=15)
    for ti in ts:
      plt.plot(ti[0],ti[1],'bx',ms=3)
    for k,gT in enumerate(gmmT):
      plt.plot(gT.mu[0],gT.mu[1],'go', ms=8.)
    plt.xlim([-0.1,2.1])
    plt.ylim([-0.1,2.1])
    plt.show()
  return ub1

def CostFunction(gmmT, t):
  c = 0.
  for i,gT in enumerate(gmmT):
    c += gT.pi * gT.pdf(t) 
  return c

import matplotlib.pyplot as plt

t = np.ones((2,1))
gmmA = [Gaussian(np.array([[0.],[0.]]), np.eye(2)*0.001, 0.3),
  Gaussian(np.array([[1.],[0.]]), np.eye(2)*0.01, 0.3),
  Gaussian(np.array([[0.],[1.]]), np.eye(2)*0.01, 0.4)]

gmmB = [Gaussian(np.array([[0.],[0.]])+t, np.eye(2)*0.001, 0.3),
  Gaussian(np.array([[1.],[0.]])+t, np.eye(2)*0.01, 0.3),
  Gaussian(np.array([[0.],[1.]])+t, np.eye(2)*0.01, 0.4)]

gmmA = [Gaussian(np.array([[0.],[0.]]), np.eye(2)*0.1, 0.3),
  Gaussian(np.array([[0.],[1.]]), np.eye(2)*0.01, 0.7)]

gmmB = [Gaussian(np.array([[0.],[0.]])+t, np.eye(2)*0.1, 0.3),
  Gaussian(np.array([[0.],[1.]])+t, np.eye(2)*0.01, 0.7)]

box = Box(np.array([[0.],[0.]]),
    np.array([[2.],[2.]]))
gmmT, A, b, Gamma_jk = ComputeGmmForT(gmmA, gmmB, \
    np.eye(2))

print(Gamma_jk)

plt.figure()
for res in [180]:
#for res in [11, 45,180]:
#for res in [10]:
  ubs = np.zeros((res,res))
  ubs2 = np.zeros((res,res))
  cs = np.zeros((res,res))
  lbs = np.zeros((res,res))
  i=res/2
  tx =  np.linspace(0,2,res)[res/2]
  #for i,tx in enumerate(np.linspace(0,2,res)):
  Ty = np.linspace(0,2,res)
  devs = np.zeros_like(Ty)
  for j,ty in enumerate(Ty):
      box = Box(np.array([[tx-1./res],[ty-1./res]]),
          np.array([[tx+1./res],[ty+1./res]]))
      ubs[i,j] = UpperBound(gmmT, box)
      ubs2[i,j] = UpperBoundConvexity(gmmT, box)
#      ubs2[i,j] = UpperBound2(gmmT, A, b, Gamma_jk, box)
      lbs[i,j] = LowerBound(gmmT, box)
      cs[i,j] = CostFunction(gmmT, box.GetMiddle())
      devs[j] = np.sqrt((box.GetMiddle()[0]-1.)**2 +
          (box.GetMiddle()[1]-1.)**2)

      
#  plt.figure()
#  plt.subplot(1,2,1)
#  plt.imshow(ubs, interpolation="nearest")
#  plt.colorbar()
#  plt.subplot(1,2,2)
#  plt.imshow(lbs, interpolation="nearest")
#  plt.colorbar()
  print(ubs2[res/2,:])

  idx = np.argsort(devs)
#
  plt.subplot(2,1,1)
  plt.plot(Ty,ubs[res/2,idx], '-', label="ub indep")
  plt.plot(Ty,ubs2[res/2,idx], '--', label="ub joint")
  plt.plot(Ty,(lbs[res/2,idx]), '-.', label="lb")
#  plt.plot(Ty,np.log(cs[res/2,:]), 'b-', label="c")
  plt.legend()
  plt.subplot(2,1,2)
  plt.plot(Ty, devs[idx])
plt.show()
