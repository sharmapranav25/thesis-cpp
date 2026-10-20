"""Three scoring arms: custom, DINO, and their calibrated mixture."""
import json
from .features.custom import Perception,Objective as CustomObjective
from .features.dino import Comparator,Objective7

def key(p):return json.dumps(p,sort_keys=True)

class Mixed:
    def __init__(self):self.custom=Perception();self.dino=Comparator()
    def extract(self,images):return self.custom.extract_pair(images)[1],self.dino.extract(images)

class Objective:
    def __init__(self,target,baseline):
        self.custom=CustomObjective(target[0],baseline[0]);self.dino=Objective7(target[1],baseline[1])
        dummy=[dict(complete=True,usable=True,strict=True)]*len(baseline[0]['whole'])
        self.calibration=[max(.05,self.custom.score(baseline[0],target[0],dummy)['appearance']),max(.05,self.dino.score(baseline[1],target[1],dummy)['appearance'])]
    def score(self,f,target,rows):
        a=self.custom.score(f[0],target[0],rows);b=self.dino.score(f[1],target[1],rows)
        ca=a['appearance']/self.calibration[0];da=b['appearance']/self.calibration[1]
        valid=4*(1-a['complete'])+2*(1-a['usable']);guard=b['contrast_penalty']
        return dict(custom_appearance=ca,dino_appearance=da,appearance=.5*(ca+da),objective=.5*(ca+da)+guard+valid,
            custom_objective=ca+valid,dino_objective=da+guard+valid,contrast_penalty=guard,complete=a['complete'],usable=a['usable'],strict=a['strict'],
            audit_pixel_distance=a['audit'],custom_detail=a,dino_detail=b)
