import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from js.utils.plot.colors import colorScheme

c1 = colorScheme("labelMap")["turquoise"]
c2 = colorScheme("labelMap")["orange"]

#paper
mpl.rc('font',size=30) 
mpl.rc('lines',linewidth=3.)
figSize = (14, 5.5)
figSize = (14, 10)
figSize = (14, 12)

def ToRad(deg):
  return deg*np.pi/180.
def ToDeg(rad):
  return rad/np.pi*180.

eta0 = ToRad(72.)
eta = ToRad(np.exp(np.linspace(np.log(0.001), np.log(180.),1000)))

a = np.cos(eta0)
b = np.cos(eta*0.5)

N = np.ceil(np.log((1./a - 1.)/(1./b - 1.))/np.log(2.))
N[N<0.] = 0.

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w",
    edgecolor="k")
ax = plt.subplot(111)
plt.plot(ToDeg(eta), N, color=c1)
#plt.plot(N, ToDeg(eta))
#ax.set_yscale("log")
ax.set_xscale("log")
plt.xlabel("desired angular precision $\eta$ [deg]")
plt.ylabel("number of required subdivisions")
plt.savefig("../angularPrecisionVsSubdivisionLvl.png", figure=fig)

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w",
    edgecolor="k")
ax = plt.subplot(111)
plt.plot(N, ToDeg(eta), color=c1)
ax.set_yscale("log")
plt.ylabel("angular precision $\eta$ [deg]")
plt.xlabel("number of subdivisions")
plt.ylim([ToDeg(eta).min(), ToDeg(eta).max()])

ax2 = ax.twinx()

r0 = 10.
r = np.exp(np.linspace(np.log(1e-10), np.log(r0),1000))
N = np.ceil(np.log(r0/r))/np.log(2.)

plt.plot(N, r, color=c2)
ax2.set_yscale("log")
plt.ylabel("translational precision $r$ [m]")
plt.ylim([r.min(), r.max()])


plt.show()
