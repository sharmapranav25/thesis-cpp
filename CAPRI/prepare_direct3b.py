from pathlib import Path
p=Path('render_direct_bank.py').read_text().split('results={};summary_panels=[]')[0].replace("'direct_bank_renders_01'","'direct_phase3b_01'")
p+='''
import time,shutil,cv2
from torch import nn
from torch.nn import functional as F
from style_pipeline.refiners import fixed_features
start=time.time();torch.manual_seed(881);cv2.setNumThreads(1)
shutil.copy2(__file__,out/'script_snapshot.py');prior=read(ROOT/'runs/style_direct_bank_01/results.json')
write(out/'protocol.json',dict(method='direct image spatial flow; no renderer parameter prediction',methods=['cnn','transformer','gan'],steps=300,selection='fixed final step; no dev selection',color='hard nearest-neighbor resampling of input RGB; bilinear straight-through gradient only',text='dilated synthetic glyph region exact; sampling from protected source prohibited outside region',max_displacement_pixels=3,resolution=[192,64],count='fixed5, no count inference',training='unpaired real fit images allowed for refiners only, no real labels',scope='exposed development diagnostic; first up to12 fit and last4 report, Pymlex2/2',automation='paused'))
names=list(families);samples=[];splits={};train=[];dev=[]
for k,(name,rows) in enumerate(families.items()):
    nfit=len(rows)-4 if len(rows)>4 else 2;devcount=len(rows)-nfit
    font=next(f['path'] for f in fonts if f['family']==prior[name]['font']);effect=prior[name]['effect'];variant=variants.get(name,0)
    assert set(r['pixel_sha256'] for r in rows[:nfit]).isdisjoint(r['pixel_sha256'] for r in rows[nfit:])
    for i,r in enumerate(rows):
        real=load(r['path']);assert pixel_hash(real)==r['pixel_sha256'];style=extract_style(real);style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()};style['font_effect']=effect
        text=''.join(np.random.default_rng(380000+i).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5));image,mask,recipe=render(style,text,font,381000+i,variant)
        image=image.resize((192,64),Image.Resampling.NEAREST);mask=mask.resize((192,64),Image.Resampling.NEAREST);real=real.resize((192,64),Image.Resampling.BILINEAR)
        idx=len(samples);samples.append(dict(family=k,path=r['path'],image=image,real=real,mask=mask,recipe=recipe));(train if i<nfit else dev).append(idx)
    splits[name]=dict(fit=[r['path'] for r in rows[:nfit]],development=[r['path'] for r in rows[nfit:]])
write(out/'splits.json',splits)
def tensor(ims):return torch.tensor(np.stack([np.asarray(im,dtype='float32').transpose(2,0,1)/255 for im in ims]),device='cuda')
x=tensor([s['image'] for s in samples]);real=tensor([s['real'] for s in samples]);mask=torch.tensor(np.stack([np.asarray(s['mask'])>0 for s in samples])[:,None],device='cuda',dtype=torch.float32);mask=F.max_pool2d(mask,5,1,2);fam=torch.tensor([s['family'] for s in samples],device='cuda');train_by=[[i for i in train if samples[i]['family']==k] for k in range(8)]
class Warp(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='transformer':
            self.embed=nn.Conv2d(12,32,8,8);self.pos=nn.Parameter(torch.zeros(1,192,32));layer=nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True);self.enc=nn.TransformerEncoder(layer,2);self.head=nn.Conv2d(32,2,3,padding=1)
        else:self.net=nn.Sequential(nn.Conv2d(12,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,24,3,padding=2,dilation=2),nn.SiLU(),nn.Conv2d(24,2,3,padding=1));self.head=self.net[-1]
        nn.init.normal_(self.head.weight,std=.001);nn.init.zeros_(self.head.bias)
    def forward(self,x,m,f):
        cond=F.one_hot(f,8).float()[:,:,None,None].expand(-1,-1,64,192);z=torch.cat([x,m,cond],1)
        if self.kind=='transformer':
            z=self.embed(z);z=self.enc(z.flatten(2).transpose(1,2)+self.pos).transpose(1,2).reshape(-1,32,8,24);raw=F.interpolate(self.head(z),size=(64,192),mode='bilinear',align_corners=False)
        else:raw=self.net(z)
        flow=3*torch.tanh(raw)*(1-m);yy,xx=torch.meshgrid(torch.linspace(-1,1,64,device=x.device),torch.linspace(-1,1,192,device=x.device),indexing='ij');grid=torch.stack([xx,yy],-1)[None]+flow.permute(0,2,3,1)*torch.tensor([2/191,2/63],device=x.device)
        soft=F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=True);hard=F.grid_sample(x,grid,mode='nearest',padding_mode='border',align_corners=True);samplemask=F.grid_sample(m,grid,mode='nearest',padding_mode='border',align_corners=True)
        protected=(m>0)|(samplemask>0);warped=hard+(soft-soft.detach());return torch.where(protected,x,warped),flow
class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(11,24,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(24,48,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(48,1,3,padding=1))
    def forward(self,x,f):return self.net(torch.cat([x,F.one_hot(f,8).float()[:,:,None,None].expand(-1,-1,64,192)],1))
def features(a):return fixed_features(a.mean(1,keepdim=True).expand(-1,3,-1,-1))
scale=features(real[train]).std(0).clamp_min(.03);history=[];outputs={}
for method in ('cnn','transformer','gan'):
    torch.manual_seed(881);rng=np.random.default_rng(881);model=Warp(method).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001);critic=Critic().cuda();co=torch.optim.Adam(critic.parameters(),lr=.0002)
    for step in range(1,301):
        ids=[int(rng.choice(group)) for group in train_by];rid=[int(rng.choice(group)) for group in train_by];xx=x[ids];mm=mask[ids];ff=fam[ids];rr=real[rid];fake,flow=model(xx,mm,ff)
        if method=='gan':
            dl=F.softplus(-critic(rr,ff)).mean()+F.softplus(critic(fake.detach(),ff)).mean();co.zero_grad();dl.backward();co.step()
            for param in critic.parameters():param.requires_grad=False
        # Unpaired same-family targets; random content must not be reconstructed.
        loss=((features(fake)-features(rr))/scale).square().mean()+.02*flow.square().mean()+.03*((flow[:,:,1:]-flow[:,:,:-1]).abs().mean()+(flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean())
        if method=='gan':loss=loss+.05*F.softplus(-critic(fake,ff)).mean()
        opt.zero_grad();loss.backward();opt.step()
        if method=='gan':
            for param in critic.parameters():param.requires_grad=True
        if step%50==0:history.append(dict(method=method,step=step,loss=float(loss.detach())));write(out/'training.json',history);print(method,step,float(loss.detach()),flush=True)
    torch.save(dict(state_dict=model.state_dict(),kind=method,step=300),out/(method+'.pt'))
    if method=='gan':torch.save(critic.state_dict(),out/'gan_critic.pt')
    model.eval()
    with torch.no_grad():output,flow=model(x[dev],mask[dev],fam[dev])
    output=(output.clamp(0,1)*255).round().byte().cpu().numpy().transpose(0,2,3,1);outputs[method]=output
    np.save(out/(method+'_flow.npy'),flow.cpu().numpy())
    for j,idx in enumerate(dev):
        original=np.asarray(samples[idx]['image']);m=mask[idx,0].cpu().numpy()>0;assert np.array_equal(output[j][m],original[m]);allowed=set(map(tuple,original.reshape(-1,3)));assert all(tuple(c) in allowed for c in np.unique(output[j].reshape(-1,3),axis=0))
        Image.fromarray(output[j]).save(out/f"{names[samples[idx]['family']]}_{method}_{j:02d}.png")
    print('validated',method,flush=True)
# Frozen DINO CLS diagnostic was not used to train any refiner.
cache=Path.home()/'.cache/torch/hub';sys.path.insert(0,str(cache/'facebookresearch_dino_main'));import vision_transformer as vit
dino=vit.vit_small(patch_size=8).cuda().eval();dino.load_state_dict(torch.load(cache/'checkpoints/dino_deitsmall8_pretrain.pth',map_location='cpu',weights_only=True))
def embed(a):
    a=F.interpolate(a,size=(72,224),mode='bilinear',align_corners=False);a=(a-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.no_grad():return F.normalize(dino(a),dim=1).cpu().numpy()
re=embed(real[dev]);emb={'baseline':embed(x[dev])};metrics={};panels=[]
for method,o in outputs.items():emb[method]=embed(torch.tensor(o.transpose(0,3,1,2)/255,dtype=torch.float32,device='cuda'))
for k,name in enumerate(names):
    positions=[j for j,idx in enumerate(dev) if samples[idx]['family']==k];ref=re[positions].mean(0);metrics[name]={}
    panel=Image.new('RGB',(1000,640),'#f1f3f5');draw=ImageDraw.Draw(panel)
    for method,e in emb.items():
        v=e[positions].mean(0);metrics[name][method]=dict(dino_distance=float(1-np.dot(v,ref)/(np.linalg.norm(v)*np.linalg.norm(ref))),changed_fraction=float(np.mean([np.any(outputs[method][j]!=np.asarray(samples[dev[j]]['image']),axis=2).mean() for j in positions])) if method!='baseline' else dict(dino_distance=float(1-np.dot(v,ref)/(np.linalg.norm(v)*np.linalg.norm(ref))))
    for i,j in enumerate(positions):
        s=samples[dev[j]];ims=[s['real'],s['image']]+[Image.fromarray(outputs[m][j]) for m in ('cnn','transformer','gan')]
        for row,im in enumerate(ims):panel.paste(im,(10+250*i,25+125*row))
    for row,label in enumerate(('Real reference','Unchanged Phase2','Phase3B CNN','Phase3B transformer','Phase3B GAN')):draw.text((10,row*125+5),name+' | '+label,fill='black')
    panel.save(out/(name+'_comparison.png'));panels.append(panel)
for k in range(2):
    board=Image.new('RGB',(1000,2560),'white')
    for i,panel in enumerate(panels[k*4:k*4+4]):board.paste(panel,(0,640*i))
    board.save(out/f'bank_{k+1}.png')
write(out/'metrics.json',metrics);write(out/'status.json',dict(state='complete',seconds=time.time()-start,fit=len(train),development=len(dev),renders=3*len(dev),exact_protected_pixels=True,no_new_rgb_values=True,count_model_used=False,promoted=False))
print('COMPLETE',flush=True)
'''
Path('run_direct3b.py').write_text(p,encoding='utf8')
