from .vMFMM import *
from .vMFgradientDescent import *
from .vMFbranchAndBound import *
from js.data.rgbd.rgbdframe import RgbdFrame
#import mayavi.mlab as mlab

if __name__ == "__main__":
  s3 = S3Grid(0)
  q = Quaternion()
#  q.setToRandom()
#  R_gt = q.toRot().R
#  print "q True: ", q.q, np.sqrt((q.q**2).sum())
  
  path = ["../data/middle_cRmf.csv", "../data/right_cRmf.csv"]
  pathRGBD = ["../data/middle_rgb", "../data/right_rgb"]
  path = ["../data/middle_cRmf.csv", "../data/left_cRmf.csv"]
  pathRGBD = ["../data/middle_rgb", "../data/left_rgb"]

  path = ["../data/middleStraightOn_cRmf.csv",
      "../data/rightStraightOn_cRmf.csv"]
  pathRGBD = ["../data/middleStraightOn_rgb",
      "../data/rightStraightOn_rgb"]

  path = ["../data/boardLevel_cRmf.csv",
      "../data/boardUp_cRmf.csv"]
  pathRGBD = ["../data/boardLevel_rgb",
      "../data/boardUp_rgb"]


  path = ["../data/middleStraightOn_cRmf.csv",
      "../data/rightStraightOn_cRmf.csv"]
  pathRGBD = ["../data/middleStraightOn_rgb",
      "../data/rightStraightOn_rgb"]

  path = ["../data/middleL50_cRmf.csv", "../data/leftL50_cRmf.csv"]
  pathRGBD = ["../data/middle_rgb", "../data/left_rgb"]

  if path is None:
    vMFs_A = [vMF(np.array([1.,0.,0.]), 1.), vMF(np.array([0.,1.,0.]), 10.)]
    vMFs_B = [vMF(R_gt.dot(np.array([1.,0.,0.])), 1.),
        vMF(R_gt.dot(np.array([0.,1.,0.])), 10.)]
    vMFMM_A = vMFMM(np.array([0.5, 0.5]), vMFs_A)
    vMFMM_B = vMFMM(np.array([0.5, 0.5]), vMFs_B)
  else:
    vMFMM_A = LoadvMFMM(path[0])
    vMFMM_B = LoadvMFMM(path[1])

  rgbdA = RgbdFrame(540.) 
  rgbdA.load(pathRGBD[0])
  rgbdB = RgbdFrame(540.) 
  rgbdB.load(pathRGBD[1])

  tetras = s3.GetTetras(0)
  tetrahedra = s3.GetTetrahedra(0)

  maxIter = 200
  fig = plt.figure()

  print("UpperBoundConvexity")
  nodes = [Node(tetrahedron) for tetrahedron in tetrahedra]
  bb = BB(vMFMM_A, vMFMM_B, LowerBoundLog, UpperBoundConvexityLog)
  epsC, q_star = bb.Compute(nodes, maxIter, q)
  
  T = bb.GetTree(0)
  T.draw("bbTree.png", prog="dot")

  figm = mlab.figure(bgcolor=(1,1,1)) 
  rgbdA.showPc(figm=figm)
  pc = rgbdB.getPc()
  for i in range(pc.shape[0]):
    for j in range(pc.shape[1]):
      pc[i,j,:] = q_star.rotate(pc[i,j,:])
  rgbdB.showPc(figm=figm)

  figm = mlab.figure(bgcolor=(1,1,1)) 
  s = Sphere(3)
  s.plotFanzy(figm,1.)
  R = q_star.toRot().R
  for j in range(vMFMM_A.GetK()):
    for k in range(vMFMM_B.GetK()):
      muA = vMFMM_A.GetvMF(j).GetMu()
      muB = vMFMM_B.GetvMF(k).GetMu()
      muB = R.dot(muB)
      mlab.points3d([muB[0]],[muB[1]],[muB[2]],
          scale_factor=0.1, color=(0,1,0), opacity = 0.5, mode="2dcircle")
      mlab.points3d([muA[0]],[muA[1]],[muA[2]],
          scale_factor=0.1, color=(1,0,0), opacity = 0.5, mode="2dcircle")
  mlab.show(stop=True)

  print("UpperBound")
  nodes = [Node(tetrahedron) for tetrahedron in tetrahedra]
  bb = BB(vMFMM_A, vMFMM_B, LowerBoundLog, UpperBoundLog)
  eps, q_star = bb.Compute(nodes, maxIter, q)

  plt.subplot(2,1,1)
  plt.ylabel("Sum over angular deviation between closest vMF means.")
  plt.plot(eps[:,:2].sum(axis=1), 'r-', label="UpperBound")
  plt.plot(epsC[:,:2].sum(axis=1), 'g-', label="UpperBoundConvexity")
  plt.legend()
  plt.subplot(2,1,2)
  plt.ylabel("Angle between GT and inferred rotation.")
  plt.plot(eps[:,3], 'r-', label="UpperBound")
  plt.plot(epsC[:,3], 'g-', label="UpperBoundConvexity")
  plt.legend()
#  plt.subplot(3,1,3)
#  plt.ylabel("Cost Function.")
#  plt.plot(eps[:,2], 'r-', label="UpperBound")
#  plt.plot(epsC[:,2], 'g-', label="UpperBoundConvexity")
#  plt.legend()

  for i in range(1):
    gd = GradientDescent(vMFMM_A, vMFMM_B)
    q.setToRandom()
    R0 = q.toRot().R
    Rt, epsGC = gd.Compute(R0, maxIter, R_gt)

    plt.subplot(2,1,1)
    plt.plot(epsGC[:,:2].sum(axis=1), 'b-' , label="Gradient")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(epsGC[:,3], 'b-', label="Gradient")
    plt.legend()
#    plt.subplot(3,1,3)
#    plt.plot(epsGC[:,2], 'b-', label="Gradient")
#    plt.legend()
  plt.savefig("./vMFMM_GradientAndBoundComparison_angleToGT.png")
  plt.savefig("./vMFMM_GradientAndBoundComparison_residuals.png")
  plt.show()

