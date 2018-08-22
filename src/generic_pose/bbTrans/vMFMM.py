import numpy as np

def LoadvMFMM(path):
  param = np.loadtxt(path)
  vMFs = []
  pi = param[4,:] / param[4,:].sum()
  print(pi)
  for k in range(param.shape[1]):
    vMFs.append(vMF(param[:3,k], param[3,k]))
    print(param[:3,k])
    print(param[3,k])
  return vMFMM(pi, vMFs)

def Compute2SinhOverZ(z):
  '''
  compute (exp(z) - exp(-z)) / z
  '''
  if np.abs(z) < 1e-6:
    return 2.
  else:
    return (np.exp(z) - np.exp(-z)) / z

def ComputeLog2SinhOverZ(z):
  '''
  compute log((exp(z) - exp(-z)) / z)
  '''
  if np.abs(z) < 1e-6:
    return np.log(2.)
  elif z < 50.:
    return - np.log(z) + np.log(np.exp(2.*z) -1) - z
  else:
    return - np.log(z) + z

def ComputeLogDeriv2SinhOverZ(z):
  '''
  compute d/dz (log((exp(z) - exp(-z)) / z))
    = ((np.exp(z)+np.exp(-z))/z) - ((np.exp(z)-np.exp(-z))/z**2)
  '''
#  if np.abs(z) < 1e-6:
#    return np.log(2.)
  if z < 50.:
    return np.log((z-1.)*np.exp(2.*z) + z + 1.) -z - 2.*np.log(z)
  else:
    return z + np.log(z-1.) + 2.*np.log(z)

class vMF(object):
  def __init__(self, mu, tau):
    self.mu = np.copy(mu)
    self.tau = tau
    self.Z = self.ComputePartitionFunction()
    self.logZ = self.ComputeLogPartitionFunction()
  def GetTau(self):
    return self.tau
  def GetMu(self):
    return self.mu
  def GetZ(self):
    return self.Z
  def GetLogZ(self):
    return self.logZ
  def ComputePartitionFunction(self):
#    return self.tau / (2.*np.pi*(np.sinh(self.tau)))
    return 1./(2.*np.pi*Compute2SinhOverZ(self.tau))
  def ComputeLogPartitionFunction(self):
    return -ComputeLog2SinhOverZ(self.tau) - np.log(2.*np.pi)

class vMFMM(object):
  def __init__(self, pis, vMFs):
    self.pis = pis / pis.sum()
    self.vMFs = vMFs
  def GetvMF(self, k):
    return  self.vMFs[k]
  def GetPi(self, k):
    return  self.pis[k]
  def GetK(self):
    return self.pis.size

def ComputevMFtovMFcost(vMFMM_A, vMFMM_B, j, k, nu):
  C = 2. * np.pi * vMFMM_A.GetPi(j) * vMFMM_B.GetPi(k) * \
      vMFMM_A.GetvMF(j).GetZ() * vMFMM_B.GetvMF(k).GetZ() 
  z_jk = np.sqrt(((vMFMM_A.GetvMF(j).GetTau() *
    vMFMM_A.GetvMF(j).GetMu() + vMFMM_B.GetvMF(k).GetTau() *
    nu)**2).sum())
  C *= Compute2SinhOverZ(z_jk)
  return C

def ComputeLogvMFtovMFcost(vMFMM_A, vMFMM_B, j, k, nu):
  C = np.log(2. * np.pi) + np.log(vMFMM_A.GetPi(j)) + \
    np.log(vMFMM_B.GetPi(k)) + vMFMM_A.GetvMF(j).GetLogZ() + \
    vMFMM_B.GetvMF(k).GetLogZ() 
  z_jk = np.sqrt(((vMFMM_A.GetvMF(j).GetTau() * \
    vMFMM_A.GetvMF(j).GetMu() + vMFMM_B.GetvMF(k).GetTau() * \
    nu)**2).sum())
  print("nu" ,nu)
  print((vMFMM_A.GetvMF(j).GetTau() * \
    vMFMM_A.GetvMF(j).GetMu() + vMFMM_B.GetvMF(k).GetTau() * \
    nu))
  print(C, z_jk, C + ComputeLog2SinhOverZ(z_jk))
  C += ComputeLog2SinhOverZ(z_jk)
  return C
