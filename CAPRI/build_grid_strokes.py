from pathlib import Path
src=Path('stroke_two_stage/train.py').read_text()
dst=Path('stroke_grid_10x3_v2');dst.mkdir(exist_ok=False)
src=src.replace(" monitor=ROOT/'monitor/strokes_two_stage_status.json';tmp=monitor.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(monitor)\n",'')
src=src.replace("bg=rng.integers(190,256,3);base=Image.new('RGB',(128,64),tuple(bg));d=ImageDraw.Draw(base)","""bg=rng.integers(0,256,3)
 arr=np.broadcast_to(bg,(64,128,3)).copy().astype(float)
 if seed%2:arr+=np.linspace(-65,65,128)[None,:,None]
 base=Image.fromarray(np.uint8(np.clip(arr,0,255)));d=ImageDraw.Draw(base)
 if seed%5==0:
  for _ in range(5):
   xx,yy=rng.integers([0,0],[110,48]);d.rectangle((int(xx),int(yy),int(xx+15),int(yy+16)),fill=tuple(rng.integers(0,256,3)))""")
src=src.replace("tuple(rng.integers(20,150,3))","tuple(rng.integers(0,256,3))")
src=src.replace("if kind:\n  t=", "if kind:\n  while np.linalg.norm(np.array(ink)-bg)<80:ink=tuple(rng.integers(0,256,3))\n  before=np.asarray(base).copy()\n  t=")
src=src.replace("return np.asarray(base),np.asarray(mask),kind", """target=np.asarray(mask).copy()
 if kind:
  delta=np.max(np.abs(np.asarray(base).astype(float)-before.astype(float)),axis=2)
  target[delta<12]=0
  assert np.count_nonzero(target)>=12,'Target must remain visible'
 return np.asarray(base),target,kind""")
start=src.index('class Detector(');end=src.index('\nclass Shape(',start)
src=src[:start]+'''def tiles(x):
 b,c,h,w=x.shape
 assert (h,w)==(64,128)
 x=F.pad(x,(0,2,0,2),mode='replicate')
 return x.reshape(b,c,3,22,10,13).permute(0,2,4,1,3,5).reshape(b*30,c,22,13)

def stitch(x,b):
 c=x.shape[1]
 return x.reshape(b,3,10,c,22,13).permute(0,3,1,4,2,5).reshape(b,c,66,130)[:,:,:64,:128]

class Detector(nn.Module):
 def __init__(self):
  super().__init__()
  self.net=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,1,1))
 def forward(self,x):
  return stitch(self.net(tiles(x)),len(x))
''' +src[end:]
src=src.replace('101000000','111000000').replace('102000000','112000000').replace('103000000','113000000')
src=src.replace("'detector_epochs':16","'view':[10,3],'feature_pop':False,'detector_epochs':24").replace('for epoch in range(16):','for epoch in range(24):',1)
src=src.replace("'one target at most, clean/plain backgrounds then text, no multi-stroke instance grouping; diagnostic not robust acceptance'","'10 columns x3 rows, shared raw RGB patch detector; stitched mask; no cross-tile context; one target maximum; colour/gradient/block variety; no multi-instance grouping'")
src=src.replace("'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()","'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()")
(dst/'train.py').write_text(src)
(dst/'PLAN.md').write_text('''# Stroke10x3v2: raw RGB multicolour trial

Explicit user request: retrain with10columns x3rows, no feature pop, multiple colours. Fresh synthetic seeds111m/112m/113m;2400 train,600 selection,600 locked images plus clean curriculum counterpart. No real fitting. Detector sees30 non-overlapping RGB patches; shared weights, not30 separate classifiers. Replicate padding extends128x64 to130x66; each patch13x22; logits stitched back and padding cropped. No extra1x1 input or postprocessing. Shape stage receives the stitched whole-image mask. Patch seams and lost long-range context are explicit risks.

Random full-range RGB backgrounds/text/strokes, both luminance polarities, colour gradients and rectangular hard negatives. Stroke colour separated from base colour; exact final visible synthetic target masks exclude negligible pixel changes. Four clean detector epochs then20 text-clutter epochs; shape16epochs. Old model/runs untouched. Synthetic masks supervise training only. Real inference uses predicted masks.

Same fixed gates: positiveIoU>=.70, absent false-positive rate<=.05, predicted-mask shape accuracy>=.85. Same threshold.5 and12pixel detection rule. Fresh font holdouts: first6train, first7selection, final2locked. No automatic promotion, no runtimeVLM, no external papers newly used. Changed architecture and colours together mean this is a requested combined experiment, not an isolated proof of grid benefit. Finite run, automation remains paused.
''')
print(dst.resolve())
