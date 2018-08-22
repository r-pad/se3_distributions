import numpy as np

def normed(a):
  return a/np.sqrt((a**2).sum())

def near(a, b):
  return np.abs(a-b) < 1e-6
  
mu1 = normed(np.array([1.,0.,0.]))
mu2 = normed(np.array([0.,1.,1.]))
nu = normed(np.array([0.,0.,1.]))

theta12 = np.arccos(mu1.dot(mu2))
theta1 = np.arccos(mu1.dot(nu))
theta2 = np.arccos(mu2.dot(nu))

t = 0.5
if near(theta1, np.pi*0.5) and near(theta2, np.pi*0.5):
  # any t is good.
  t = 0.5
t = (np.arctan2(np.cos(theta2) - np.cos(theta12)*np.cos(theta1),np.cos(theta1)*np.sin(theta12))) / theta12

print(theta1, theta2, theta12, t)
