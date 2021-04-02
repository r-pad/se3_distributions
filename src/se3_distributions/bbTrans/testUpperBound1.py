import numpy as np

def normed(a):
  return a/np.sqrt((a**2).sum())

def LogS2(p, q):
  theta = np.arccos(mu1.dot(mu2))
  if theta < 1e-6:
    return q-p
  else:
    return (q - np.cos(theta)*p) * theta / np.sin(theta)
  
mu1 = normed(np.array([1.,0.,0.]))
mu2 = normed(np.array([0.,1.,0.]))
nu = normed(np.array([0.,1.,0.]))

theta = np.arccos(mu1.dot(mu2))
x = LogS2(mu1, mu2)

t = 0.9
dt = 999.
while np.abs(dt) > 1e-12:
  f = x.dot(mu1) - 2. + 2./(theta*t)**2 - 1./np.tan(theta*t)
  df = (1./t)*(1./(np.tan(theta*t)**2)) - 4./(t**3 * theta**2)
  dt = - f/df
  t = t + dt
  print(f, df, dt)

print(t, t*theta * 180./np.pi)
