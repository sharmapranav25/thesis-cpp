from pathlib import Path
p=Path('render_direct_bank.py').read_text().split('results={};summary_panels=[]')[0].replace("'direct_bank_renders_01'","'family_phase3b_01'").replace("['fit'][:16]","['fit'][:96]")
p+='''
import time,shutil,cv2
from torch import nn
from torch.nn import functional as F
from style_pipeline.refiners import fixed_features
start=time.time();cv2.setNumThreads(1);prior=read(ROOT/'runs/style_direct_bank_01/results.json')
shutil.copy2(__file__,out/'script_snapshot.py')
write(out/'protocol.json',dict(method='eight independently initialized family-specific GAN pixel refiners; no shared trained weights, no family conditioning',training='only own-family real/synthetic fit pixels; unpaired, no real text labels',max_steps=1500,min_steps=500,validation_every=100,patience=4,selection='lowest own-family validation grayscale moment distance; identity baseline eligible',pixels='RGB residual projected to own synthetic input color candidates; straight-through gradient; original pixel also candidate',glyphs='exact dilated mask protection',count='fixed5; count inference disabled',split='96 references where available:72fit/12selection/12report; small sources proportional, pymlex2/1/1',limitations='previously exposed development; frozen upstream font choices previously exposed; tiny sources insufficient distribution evidence',automation='paused'))
class Generator(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=2,dilation=2),nn.SiLU(),nn.Conv2d(32,32,3,padding=4,dilation=4),nn.SiLU(),nn.Conv2d(32,3,3,padding=1));nn.init.normal_(self.net[-1].weight,std=.01);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,m,palette):
        proposed=(x+.35*torch.tanh(self.net(torch.cat([x,m],1)))).clamp(0,1);flat=proposed.flatten(2).transpose(1,2);dist=(flat*flat).sum(2,keepdim=True)+(palette*palette).sum(2)[:,None]-2*torch.bmm(flat,palette.transpose(1,2));values,ids=dist.min(2);chosen=palette.gather(1,ids[:,:,None].expand(-1,-1,3)).transpose(1,2).reshape_as(x);origdist=((proposed-x)**2).sum(1);hard=torch.where((values.reshape_as(origdist)<origdist)[:,None],chosen,x);y=hard+(proposed-proposed.detach());return torch.where(m.bool(),x,y)
class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(3,24,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(24,48,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(48,1,3,padding=1))
    def forward(self,x):return self.net(x)
def feats(x):return fixed_features(x.mean(1,keepdim=True).expand(-1,3,-1,-1))
def tensor(images):return torch.tensor(np.stack([np.asarray(im,dtype='float32').transpose(2,0,1)/255 for im in images]),device='cuda')
results={};allpanels=[]
for familyindex,(name,rows) in enumerate(families.items()):
    familyout=out/name;familyout.mkdir();torch.manual_seed(491+familyindex);rng=np.random.default_rng(491+familyindex)
    n=len(rows);nfit=72 if n>=96 else max(2,int(n*.75));nselect=max(1,(n-nfit)//2);partitions={'fit':rows[:nfit],'selection':rows[nfit:nfit+nselect],'report':rows[nfit+nselect:]};assert all(partitions.values())
    hashes=[r['pixel_sha256'] for r in rows];assert len(set(hashes))==len(hashes)
    write(familyout/'splits.json',{k:[dict(path=r['path'],pixel_sha256=r['pixel_sha256']) for r in rs] for k,rs in partitions.items()})
    font=next(f['path'] for f in fonts if f['family']==prior[name]['font']);effect=prior[name]['effect'];variant=variants.get(name,0);data={}
    for partition,rs in partitions.items():
        inputs=[];masks=[];targets=[];palettes=[];records=[]
        for i,r in enumerate(rs):
            real=load(r['path']);assert pixel_hash(real)==r['pixel_sha256'];style=extract_style(real);style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()};style['font_effect']=effect
            for replica in range(2 if partition=='fit' else 1):
                seed=490000+familyindex*10000+list(partitions).index(partition)*2000+i*2+replica;text=''.join(np.random.default_rng(seed).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5));im,mask,recipe=render(style,text,font,seed,variant);im=im.resize((192,64),Image.Resampling.NEAREST);mask=mask.resize((192,64),Image.Resampling.NEAREST)
                arr=np.asarray(im);colors,counts=np.unique(arr.reshape(-1,3),axis=0,return_counts=True);order=np.argsort(counts)[::-1];frequent=colors[order[:16]];spread=colors[np.linspace(0,len(colors)-1,16).astype(int)];palette=np.concatenate([frequent,spread]);palette=np.concatenate([palette,np.repeat(palette[:1],32-len(palette),axis=0)])
                inputs.append(im);masks.append(np.asarray(mask)>0);targets.append(real.resize((192,64),Image.Resampling.BILINEAR));palettes.append(palette);records.append(dict(reference=r['path'],seed=seed,text=text,recipe=recipe))
        data[partition]=dict(x=tensor(inputs),real=tensor(targets),mask=F.max_pool2d(torch.tensor(np.stack(masks)[:,None],device='cuda',dtype=torch.float32),5,1,2),palette=torch.tensor(np.stack(palettes)/255,device='cuda',dtype=torch.float32),inputs=inputs,targets=targets)
        write(familyout/(partition+'_synthetic_records.json'),records)
    g=Generator().cuda();d=Critic().cuda();go=torch.optim.Adam(g.parameters(),lr=.0005);do=torch.optim.Adam(d.parameters(),lr=.0002);fit=data['fit'];val=data['selection'];scale=feats(fit['real']).std(0).clamp_min(.03)
    def distance(fake,real):
        a=feats(fake)/scale;b=feats(real)/scale;return (a.mean(0)-b.mean(0)).square().mean()+.1*(a.std(0,unbiased=False)-b.std(0,unbiased=False)).square().mean()
    with torch.no_grad():baseline=float(distance(val['x'],val['real']))
    best=baseline;beststep=0;stale=0;history=[];torch.save(dict(identity=True,step=0),familyout/'selected.pt')
    for step in range(1,1501):
        ids=rng.integers(len(fit['x']),size=8);rids=rng.integers(len(fit['real']),size=8);x=fit['x'][ids];m=fit['mask'][ids];pal=fit['palette'][ids];rr=fit['real'][rids];fake=g(x,m,pal)
        dl=F.softplus(-d(rr)).mean()+F.softplus(d(fake.detach())).mean();do.zero_grad();dl.backward();do.step()
        for param in d.parameters():param.requires_grad=False
        loss=distance(fake,rr)+.1*F.softplus(-d(fake)).mean()+.1*(fake-x).abs().mean();go.zero_grad();loss.backward();go.step()
        for param in d.parameters():param.requires_grad=True
        if step%100==0:
            g.eval()
            with torch.no_grad():pred=g(val['x'],val['mask'],val['palette']);v=float(distance(pred,val['real']));change=float((pred-val['x']).abs().mean())
            history.append(dict(step=step,train_loss=float(loss.detach()),critic_loss=float(dl.detach()),validation=v,mean_pixel_change=change));write(familyout/'training.json',history)
            if v<best-1e-5:best=v;beststep=step;stale=0;torch.save(dict(identity=False,step=step,state_dict=g.state_dict()),familyout/'selected.pt')
            else:stale+=1
            print(name,step,'validation',round(v,4),'best',beststep,flush=True);g.train()
            if step>=500 and stale>=4:break
    torch.save(g.state_dict(),familyout/'last_generator.pt');torch.save(d.state_dict(),familyout/'last_critic.pt');selected=torch.load(familyout/'selected.pt',weights_only=True);g.eval();report=data['report']
    if not selected['identity']:g.load_state_dict(selected['state_dict'])
    with torch.no_grad():output=report['x'] if selected['identity'] else g(report['x'],report['mask'],report['palette']);before=float(distance(report['x'],report['real']));after=float(distance(output,report['real']))
    arr=(output*255).round().clamp(0,255).byte().cpu().numpy().transpose(0,2,3,1);changed=[]
    panel=Image.new('RGB',(1000,390),'#f1f3f5');draw=ImageDraw.Draw(panel)
    for i,(a,im,realim) in enumerate(zip(arr,report['inputs'],report['targets'])):
        original=np.asarray(im);mask=report['mask'][i,0].cpu().numpy()>0;assert np.array_equal(a[mask],original[mask]);allowed=set(map(tuple,original.reshape(-1,3)));assert all(tuple(c) in allowed for c in np.unique(a.reshape(-1,3),axis=0));changed.append(float(np.any(a!=original,axis=2).mean()));im.save(familyout/f'{i:02d}_before.png');Image.fromarray(a).save(familyout/f'{i:02d}_after.png')
        if i<4:
            for j,image in enumerate([realim,im,Image.fromarray(a)]):panel.paste(image,(10+i*250,25+125*j))
    for j,label in enumerate(('Real report reference','Unchanged Phase2','Own-family GAN; selected step '+str(beststep))):draw.text((10,5+j*125),name+' | '+label,fill='black')
    panel.save(familyout/'comparison.png');allpanels.append(panel)
    results[name]=dict(real_fit=len(partitions['fit']),real_selection=len(partitions['selection']),real_report=len(partitions['report']),synthetic_fit=len(fit['x']),steps_trained=step,selected_step=beststep,validation_baseline=baseline,validation_best=best,report_before=before,report_after=after,mean_changed_fraction=float(np.mean(changed)),character_pixels_exact=True,no_new_rgb=True,checkpoint_sha256=digest(familyout/'selected.pt'));write(out/'results.json',results)
    write(out/'status.json',dict(state='running',completed_families=list(results),current_family=name));print('FINISHED',name,flush=True)
for k in range(2):
    board=Image.new('RGB',(1000,1560),'white')
    for i,panel in enumerate(allpanels[k*4:k*4+4]):board.paste(panel,(0,i*390))
    board.save(out/f'bank_{k+1}.png')
write(out/'status.json',dict(state='complete',seconds=time.time()-start,completed_families=list(results),shared_weights=False,promoted=False));print('COMPLETE',flush=True)
'''
Path('run_family3b.py').write_text(p,encoding='utf8')
