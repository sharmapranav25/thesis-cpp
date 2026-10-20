"""Post-training location audit on the fixed ordinary synthetic test."""
from experiment import *
protocol=json.loads((ROOT/'protocol.json').read_text());fonts=json.loads((FONTROOT/'fonts.json').read_text());fonts=[f for f in fonts if f['family'] in protocol['train_fonts']]
# Preserve original shuffled font order, otherwise the regenerated pixels differ.
byname={f['family']:f for f in fonts};fonts=[byname[n] for n in protocol['train_fonts']]
x,targets,meta=dataset(fonts,440,41000000,False)
expected=json.loads((ROOT/'pixel_hashes.json').read_text())['test']
assert [hashlib.sha256(im.numpy().tobytes()).hexdigest() for im in x]==expected
results=json.loads((ROOT/'results.json').read_text());report={};device='cuda' if torch.cuda.is_available() else 'cpu'
def iou(a,b):
 al=a[:,:2]-a[:,2:]/2;ar=a[:,:2]+a[:,2:]/2;bl=b[:,:2]-b[:,2:]/2;br=b[:,:2]+b[:,2:]/2
 inter=np.maximum(0,np.minimum(ar[:,None],br[None])-np.maximum(al[:,None],bl[None])).prod(2)
 return inter/(a[:,2:].prod(1)[:,None]+b[:,2:].prod(1)[None]-inter+1e-9)
for seed in protocol['seeds']:
 net=Net(True).to(device);net.load_state_dict(torch.load(ROOT/f'instance_transformer_{seed}.pt',map_location=device,weights_only=True));net.eval();th=results[f'instance_transformer_{seed}']['object_threshold'];tp=predtotal=truth=0
 with torch.inference_mode():
  for start in range(0,len(x),64):
   c,o,b=net(x[start:start+64].to(device));obj=o.sigmoid().cpu().numpy();boxes=b.cpu().numpy()
   for j,(probs,bb) in enumerate(zip(obj,boxes)):
    chosen=bb[probs>=th];target=targets[start+j].numpy();predtotal+=len(chosen);truth+=len(target)
    if len(chosen) and len(target):
     matrix=iou(chosen,target);a,b=linear_sum_assignment(-matrix);tp+=int((matrix[a,b]>=.5).sum())
 report[str(seed)]=dict(iou_threshold=.5,true_positive=tp,predicted=predtotal,truth=truth,precision=tp/max(1,predtotal),recall=tp/max(1,truth),scope='ordinary locked synthetic test,440images')
save('localization_metrics.json',report);print(json.dumps(report,indent=2))
