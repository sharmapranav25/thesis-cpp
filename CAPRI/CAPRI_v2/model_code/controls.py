"""Exact procedural targets, normalized; no real labels or filename decoding."""
import numpy as np
NAMES=['line_count','line_width','curve_count','curve_width','curve_amplitude','dot_count','dot_radius','grid_spacing','blur_radius']
OWNERS=[0,0,1,1,1,2,2,4,12]
LOW=np.array([1,1,1,1,9,8,1,9,.8],np.float32)
HIGH=np.array([4,3,4,3,24,64,3,26,1.5],np.float32)
def targets(seed,part):
 values=[]
 for k in [0,1,2]:
  r=np.random.default_rng(np.random.SeedSequence([int(seed),k+1]));n=int(r.integers(8,65) if k==2 else r.integers(1,5));widths=[];amps=[];radii=[]
  for _ in range(n):
   r.integers(0,256,3);widths.append(int(r.integers(1,4)))
   if k==0:r.integers([0,0],[192,96],(2,2))
   elif k==1:
    r.uniform(20,76);amps.append(r.uniform(9,24))
    if part!='locked_recipe':r.uniform(3,7);r.uniform(-2,2)
   else:r.integers([0,0],[192,96]);radii.append(int(r.integers(1,4)))
  values.extend([n,np.mean(radii) if k==2 else np.mean(widths)])
  if k==1:values.append(np.mean(amps))
 r=np.random.default_rng(np.random.SeedSequence([int(seed),5]));r.integers(0,256,3);values.append(int(r.integers(9,27)))
 r=np.random.default_rng(np.random.SeedSequence([int(seed),13]));values.append(r.uniform(.8,1.5))
 return (np.array(values,np.float32)-LOW)/(HIGH-LOW)
def decode(v):return dict(zip(NAMES,map(float,LOW+np.clip(v,0,1)*(HIGH-LOW))))
