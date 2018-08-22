
def distinguishableColors():
  import numpy as np
  import os
  return np.loadtxt(os.path.dirname(os.path.realpath(__file__))+"/distinguishableColors.csv")

def colorScheme(name):
  rgbColor = []
  rgbColor.append((232/255.0,65/255.0,32/255.0)) # red         
  rgbColor.append((32/255.0,232/255.0,59/255.0)) # green       
  rgbColor.append((32/255.0,182/255.0,232/255.0)) # tuerkis    
  
  labelColor = []
  labelColor.append((32/255.0,182/255.0,232/255.0)) # tuerkis    
  labelColor.append((232/255.0,139/255.0,32/255.0)) # orange     
  labelColor.append((255/255.0,13/255.0,255/255.0)) # pink       
  labelColor.append((32/255.0,232/255.0,59/255.0)) # green       
  labelColor.append((232/255.0,65/255.0,32/255.0)) # red         
  labelColor.append((32/255.0,65/255.0,255/255.0)) # blue
  labelColor.append((201/255.0,189/255.0,79/255.0)) # yellow
  labelColor.append((132/255.0,73/255.0,194/255.0)) # purple
  labelColor.append((199/255.0,153/255.0,132/255.0)) # brown

  labelColorMap = dict()
  labelColorMap['red'] = (232/255.0,65/255.0,32/255.0) # red         
  labelColorMap['pink'] = (255/255.0,13/255.0,255/255.0) # pink       
  labelColorMap['orange'] = (232/255.0,139/255.0,32/255.0) # orange     
  labelColorMap['turquoise'] = (32/255.0,182/255.0,232/255.0) # tuerkis    
  labelColorMap['yellow'] = (255/255.0,255/255.0,0/255.0) # yellow
  labelColorMap['green'] = (32/255.0,232/255.0,59/255.0) # green       
  labelColorMap['blue'] = (32/255.0,65/255.0,255/255.0) # blue

  if name=='rgb':
    return rgbColor
  elif name=='label':
    return labelColor
  elif name=='labelMap':
    return labelColorMap
  else:
    print('color not found')
    return []

