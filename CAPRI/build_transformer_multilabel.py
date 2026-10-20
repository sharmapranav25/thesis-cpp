from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth');dst=Path('transformer_multilabel_v1');dst.mkdir(exist_ok=False)
shutil.copy2(root/'feature_multilabel_v1/corpus.py',dst/'corpus.py')
s=(root/'feature_multilabel_v1/train.py').read_text()
a=s.index('class Model(');b=s.index('\ndef ap(',a)
s=s[:a]+'''class Model(nn.Module):
 def __init__(self):
  super().__init__();dim=96
  self.patch=nn.Linear(3*8*8,dim)
  self.position=nn.Parameter(torch.zeros(1,12*24,dim))
  self.queries=nn.Parameter(torch.zeros(1,len(FEATURES),dim))
  layer=nn.TransformerEncoderLayer(d_model=dim,nhead=4,dim_feedforward=384,dropout=.2,activation='gelu',batch_first=True,norm_first=True)
  self.encoder=nn.TransformerEncoder(layer,num_layers=3,norm=nn.LayerNorm(dim),enable_nested_tensor=False)
  # PyTorch clones layer values; initialize each layer independently.
  for block in self.encoder.layers:
   for p in block.parameters():
    if p.ndim>1:nn.init.xavier_uniform_(p)
  self.dropout=nn.Dropout(.2);self.heads=nn.ModuleList([nn.Linear(dim,1) for _ in FEATURES])
  nn.init.trunc_normal_(self.position,std=.02);nn.init.trunc_normal_(self.queries,std=.02)
 def forward(self,x):
  b,c,h,w=x.shape
  assert (h,w)==(96,192)
  patches=x.reshape(b,c,12,8,24,8).permute(0,2,4,1,3,5).reshape(b,288,192)
  tokens=self.dropout(self.patch(patches)+self.position)
  encoded=self.encoder(torch.cat([self.queries.expand(b,-1,-1),tokens],1))
  return torch.cat([head(encoded[:,i]) for i,head in enumerate(self.heads)],1)
''' +s[b:]
s=s.replace("GRIDS=[(1,1),(2,2),(5,3),(10,3)]","GRIDS=[(24,12)]")
for old,new in [('204000000','304000000'),('205000000','305000000'),('206000000','306000000'),('207000000','307000000')]:s=s.replace(old,new)
s=s.replace('multilabel_status.json','transformer_multilabel_status.json').replace('V2_MULTILABEL_RESULTS.md','V2_TRANSFORMER_MULTILABEL_RESULTS.md').replace('Multi-label v1','Transformer multi-label v1')
s=s.replace("views=GRIDS,feature_pop=False","views=GRIDS,architecture='pure patch transformer:8x8patches,96dim,3layers,4heads,15learned feature tokens',feature_pop=False")
needle=" passed={f:all(results[str(s)][part][f]['f1']"
pos=s.index(needle)
s=s[:pos]+''' # Frozen CNN replay on exactly the fresh test images, with original calibration thresholds.
 import importlib.util
 spec=importlib.util.spec_from_file_location('cnn_comparison',ROOT/'feature_multilabel_v1/train.py');cnn=importlib.util.module_from_spec(spec);spec.loader.exec_module(cnn)
 old=json.loads((ROOT/'feature_multilabel_v1/run/results.json').read_text());baseline={}
 for seed in protocol['seeds']:
  net=cnn.Model().cuda();net.load_state_dict(torch.load(ROOT/f'feature_multilabel_v1/run/model_{seed}.pt',weights_only=True));threshold=[old['results'][str(seed)]['thresholds'][f] for f in FEATURES];baseline[str(seed)]={}
  for part in ['locked_same','locked_font','locked_combinations','locked_recipe']:
   baseline[str(seed)][part]=metrics(*predict(net,part),threshold);del cache[part]
  del net
 write('cnn_fresh_comparison.json',baseline)
''' +s[pos:]
(dst/'train.py').write_text(s)
print(dst.resolve())
