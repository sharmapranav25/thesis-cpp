"""Create isolated dot-specialist code; never edit the running curve worker."""
from pathlib import Path
import shutil
base=Path(__file__).parent/'feature_specialists';out=Path(__file__).parent/'dot_specialist';out.mkdir(exist_ok=True)
s=(base/'corpus.py').read_text()
for n in range(81,88):s=s.replace(str(n)+'000000',str(n+10)+'000000')
a=s.index('def pop(');b=s.index('\ndef draw_mask',a)
s=s[:a]+'''def pop(images):
    result=[]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    for im in images:
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        bright=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)
        dark=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
        result.append(np.maximum(bright,dark))
    return np.stack(result)
''' + s[b:]
s=s.replace("        for _ in range(int(rng.integers(0,160))):d.point(tuple(rng.integers([0,0],[128,64])),fill=tuple(rng.integers(0,256,3)))\n",'')
a=s.index("        t=np.linspace(-1,1,100)");b=s.index("        ink=tuple",a)
s=s[:a]+'''        # Dot positives versus compact dash negatives; no scatter in common backdrop.
        count=int(rng.integers(6,49));radius=int(rng.integers(1,4));recipe='clustered_disks' if part=='locked_shape' else 'uniform_disks'
        raster='opencv' if part=='locked_raster' else 'pillow';cm=np.zeros((64,128),np.uint8);sm=np.zeros_like(cm)
        circles=Image.fromarray(cm);lines=Image.fromarray(sm);dc=ImageDraw.Draw(circles);dl=ImageDraw.Draw(lines)
        cluster=rng.uniform([12,12],[116,52]);centers=[]
        for _ in range(count):
            center=(np.clip(cluster+rng.normal(0,[13,8]),[4,4],[123,59]) if part=='locked_shape' else rng.uniform([4,4],[123,59]));x,y=np.rint(center).astype(int);centers.append([int(x),int(y)])
            half=max(2,round(1.5*radius));angle=float(rng.uniform(0,np.pi));dx,dy=np.cos(angle)*half,np.sin(angle)*half;ends=[(round(x-dx),round(y-dy)),(round(x+dx),round(y+dy))]
            if raster=='opencv':cv2.circle(cm,(int(x),int(y)),radius,255,-1,cv2.LINE_8);cv2.line(sm,ends[0],ends[1],255,max(1,radius),cv2.LINE_8)
            else:dc.ellipse((x-radius,y-radius,x+radius,y+radius),fill=255);dl.line(ends,fill=255,width=max(1,radius))
        if raster=='pillow':cm=np.asarray(circles);sm=np.asarray(lines)
        negative_kind='dash' if rng.random()<.8 else 'no_added_marks'
        if negative_kind=='no_added_marks':sm=np.zeros_like(sm)
''' + s[b:]
s=s.replace("        if not .75<=(cm>0).sum()/max(1,(sm>0).sum())<=1.33:continue\n",'')
s=s.replace("if visible.sum()<24 or fraction<.4 or negvis.sum()<24 or negvis.sum()/max(1,(sm>0).sum())<.4:continue", "if visible.sum()<24 or fraction<.4:continue\n        if negative_kind=='dash' and (negvis.sum()<24 or negvis.sum()/max(1,(sm>0).sum())<.4):continue")
s=s.replace("width=width,font=", "radius=radius,count=count,negative_kind=negative_kind,centers=centers,font=")
s=s.replace("curve_bend=bend,",'').replace('curve_pixels=support','dot_pixels=support').replace("Failed valid curve pair", "Failed valid dot pair")
s=s.replace(" and r['curve_bend']>=2",'')
s=s.replace("angle_bin=seed%12;theta=(angle_bin+rng.uniform(.05,.95))*math.pi/12", "density_bin=seed%12")
s=s.replace("count=int(rng.integers(6,49));radius", "count=int(rng.integers(6+(density_bin*43)//12,6+((density_bin+1)*43)//12));radius")
s=s.replace("angle_bin=angle_bin,axis_degrees=theta*180/math.pi,", "density_bin=density_bin,")
s=s.replace("r['angle_bin']", "r['density_bin']").replace('orientation_counts','density_counts')
(out/'corpus.py').write_text(s,encoding='utf8')
t=(base/'train_curves.py').read_text().replace('curve','dot').replace('Curve','Dot').replace('CURVE','DOT')
# Independent monitor card and run files: do not contend for the old singleton dashboard.
t=t.replace("            sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update\n            update('dot_specialist_'+stage,json.dumps(values),running=stage not in ('complete','failed','prepared'),run=str(out))", "            pass  # monitor/specialists.html reads each worker's own atomic status file")
t=t.replace('D084 outcome','D085 outcome').replace('No worker remaining; other feature specialists queued.','Dot worker completed; check curve-worker status separately.')
t=t.replace("['angle_bin']", "['density_bin']").replace("['angle_bins']", "['density_bins']")
(out/'train_dots.py').write_text(t,encoding='utf8')
print(out)
