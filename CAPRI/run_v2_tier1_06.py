"""Local stroke geometry + neighboring-cell relations; synthetic-only pilot."""
from pathlib import Path
import json,time,shutil
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
import run_v2_tier1_04 as old
base=old.base;GRIDS=old.GRIDS;HERE=Path(__file__).parent;OUT=HERE/'v2_tier1_06'

def topology():
    bounds=[];edges=[];offset=0
    for cols,rows in GRIDS:
        for j in range(rows):
            for i in range(cols):
                bounds.append((int(np.ceil(i/cols*127)),int(np.ceil(j/rows*63)),int(np.floor((i+1)/cols*127))+1,int(np.floor((j+1)/rows*63))+1))
                if i+1<cols:edges.append((offset+j*cols+i,offset+j*cols+i+1))
                if j+1<rows:edges.append((offset+j*cols+i,offset+(j+1)*cols+i))
        offset+=cols*rows
    return bounds,edges
BOUNDS,EDGES=topology()

def geometry_targets(masks):
    """Native-coordinate targets from visible line/curve union, no parent-type label."""
    targets=[];links=[]
    for masks_one in masks:
        union=np.maximum(masks_one[0],masks_one[1]);binary=np.uint8(union>=.25)
        _,components=cv2.connectedComponents(binary,connectivity=8)
        nodes=[];members=[]
        for x0,y0,x1,y1 in BOUNDS:
            patch=union[y0:y1,x0:x1];yy,xx=np.where(patch>=.25);n=len(xx)
            node=np.zeros(7,np.float32);node[6]=float(patch.mean())
            if n>=3:
                node[0]=1;node[1:3]=[float(xx.mean()/max(1,x1-x0-1)),float(yy.mean()/max(1,y1-y0-1))]
                xy=np.stack([xx,yy],1).astype(float);cov=np.cov(xy,rowvar=False);vals,vecs=np.linalg.eigh(cov);axis=vecs[:,-1]
                node[3:5]=[axis[0]**2-axis[1]**2,2*axis[0]*axis[1]]
                labels=components[y0:y1,x0:x1][yy,xx];_,counts=np.unique(labels,return_counts=True)
                node[5]=float((vals[-1]-vals[0])/max(vals.sum(),1e-6)*counts.max()/n)
                ids,counts=np.unique(labels,return_counts=True);members.append(set(ids[counts>=3].tolist())-{0})
            else:members.append(set())
            nodes.append(node)
        targets.append(nodes);links.append([float(bool(members[a]&members[b])) for a,b in EDGES])
    return np.asarray(targets,np.float32),np.asarray(links,np.float32)

class GeometryModel(base.Multi):
    def __init__(self,relations=True):
        super().__init__(True);self.relations=relations
        self.geometry=nn.ModuleList([nn.Sequential(nn.Linear(48,32),nn.ReLU(),nn.Linear(32,7)) for _ in GRIDS])
        self.node_embed=nn.Sequential(nn.Linear(59,48),nn.ReLU())
        self.edge_embed=nn.Sequential(nn.Linear(19,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.link=nn.Linear(32,1);self.message=nn.Linear(32,48)
        self.node_score=nn.Linear(48,1);self.grid_score=nn.Linear(48,1)
        self.register_buffer('edges',torch.tensor(EDGES,dtype=torch.long))
    def forward(self,x):
        b=len(x);v=len(self.grid)
        views=F.grid_sample(x[:,None].expand(-1,v,-1,-1,-1).reshape(b*v,3,64,128),self.grid[None].expand(b,-1,-1,-1,-1).reshape(b*v,24,48,2),align_corners=True)
        tokens=self.encoder(views).mean((2,3)).reshape(b,v,48);raw=[];offset=0
        for head,(cols,rows) in zip(self.geometry,GRIDS):
            n=cols*rows;raw.append(head(tokens[:,offset:offset+n]));offset+=n
        raw=torch.cat(raw,1);g=torch.cat([raw[:,:,:3].sigmoid(),F.normalize(raw[:,:,3:5],dim=2,eps=1e-6),raw[:,:,5:].sigmoid()],2)
        boxes=self.boxes[None].expand(b,-1,-1);nodes=self.node_embed(torch.cat([tokens,boxes,g],2))
        u,w=self.edges[:,0],self.edges[:,1];a,c=g[:,u],g[:,w]
        centers=boxes[:,:,:2]+g[:,:,1:3]*boxes[:,:,2:];delta=centers[:,w]-centers[:,u]
        dot=(a[:,:,3:5]*c[:,:,3:5]).sum(2,keepdim=True);cross=(a[:,:,3:4]*c[:,:,4:5]-a[:,:,4:5]*c[:,:,3:4])
        e=self.edge_embed(torch.cat([a,c,delta,dot,cross,delta.norm(dim=2,keepdim=True)],2));link=self.link(e).squeeze(2)
        if self.relations:
            # Predicted evidence only: no target masks/tilts/connections enter forward.
            gate=(link.sigmoid()*a[:,:,0]*c[:,:,0]*torch.sqrt(a[:,:,5]*c[:,:,5]+1e-8))[:,:,None]
            messages=self.message(e)*gate;aggregate=torch.zeros_like(nodes);aggregate.index_add_(1,u,messages);aggregate.index_add_(1,w,messages)
            degree=torch.zeros(v,device=x.device);degree.index_add_(0,u,torch.ones_like(u,dtype=x.dtype));degree.index_add_(0,w,torch.ones_like(w,dtype=x.dtype))
            nodes=nodes+aggregate/degree.clamp_min(1)[None,:,None]
        pools=[];offset=0
        for cols,rows in GRIDS:
            n=cols*rows;part=nodes[:,offset:offset+n];pools.append((self.node_score(part).softmax(1)*part).sum(1));offset+=n
        pools=torch.stack(pools,1);context=(self.grid_score(pools).softmax(1)*pools).sum(1)
        spatial=self.encoder(x)+self.context(context)[:,:,None,None];detail=self.detail(x)
        mid=F.interpolate(self.mid(detail),size=x.shape[-2:],mode='bilinear',align_corners=False);coarse=F.interpolate(spatial,size=x.shape[-2:],mode='bilinear',align_corners=False)
        z=self.decode(torch.cat([detail,mid,coarse],1));prob=z.sigmoid().flatten(2);summary=torch.cat([prob.mean(2),prob.topk(32,dim=2).values.mean(2)],1)
        self.raw_geometry=raw;self.last_geometry=g;self.last_links=link
        return self.presence(torch.cat([context,summary],1)),z

def local_loss(model,t,l):
    raw=model.raw_geometry;g=model.last_geometry;present=t[:,:,0];valid=present*(t[:,:,5]>=.6);losses=[];offset=0
    for cols,rows in GRIDS:
        n=cols*rows;s=slice(offset,offset+n);p=present[:,s];v=valid[:,s]
        presence=F.binary_cross_entropy_with_logits(raw[:,s,0],p)
        center=((g[:,s,1:3]-t[:,s,1:3]).abs().mean(2)*p).sum()/p.sum().clamp_min(1)
        direction=((1-(g[:,s,3:5]*t[:,s,3:5]).sum(2))*v).sum()/v.sum().clamp_min(1)
        confidence=F.mse_loss(g[:,s,5],t[:,s,5]);density=F.mse_loss(g[:,s,6],t[:,s,6])
        losses.append(presence+center+direction+confidence+density);offset+=n
    return torch.stack(losses).mean()+F.binary_cross_entropy_with_logits(model.last_links,l)

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1);start=time.time()
    for source in (__file__,old.__file__,base.__file__):shutil.copy2(source,OUT/Path(source).name)
    fonts=json.loads((HERE/'v2_tier1_03/fonts.json').read_text());base.write(OUT/'fonts.json',fonts)
    base.write(OUT/'protocol.json',dict(grids=GRIDS,crops=63,edges=len(EDGES),local_targets=['stroke presence','center x/y','cos2theta/sin2theta','coherence proxy','visible coverage'],training='synthetic only; no local parent line/curve labels',geometry='PCA on native-pixel visible union; axial angle; direction supervised only coherence>=.6',connectivity='common visible connected-component IDs with>=3pixels in each neighboring block; proxy not instance identity',comparators=['geometry_without_neighbors','geometry_with_neighbors','frozen03','frozen04'],steps=1800,batch=12,seeds=[71000000,72000000,73000000],selection='synthetic global BCE+dice; same rule as04',limitations='single seed/procedural generator; target coherence is geometric not calibrated uncertainty; shared components may merge intersections; no acceptance gate'))
    data={}
    for part in ('train','selection'):
        d=np.load(HERE/'v2_tier1_03'/(part+'_supervision.npz'));m=d['visible_masks'].astype(np.float32)/255;g,l=geometry_targets(m)
        np.savez_compressed(OUT/(part+'_geometry.npz'),geometry=g,links=l)
        data[part]=tuple(torch.tensor(a,dtype=torch.float32,device='cuda') for a in (d['images'].transpose(0,3,1,2)/255,m,d['labels'],g,l));print('BUILT GEOMETRY',part,flush=True)
    pw=(1-data['train'][2]).sum(0)/data['train'][2].sum(0).clamp_min(1);mw=((1-data['train'][1]).sum((0,2,3))/data['train'][1].sum((0,2,3)).clamp_min(1)).clamp(1,20)[None,:,None,None]
    def global_loss(a,z,y,m):
        p=z.sigmoid();dice=1-(2*(p*m).sum((2,3))+1)/((p+m).sum((2,3))+1)
        return 3*F.binary_cross_entropy_with_logits(a,y,pos_weight=pw)+F.binary_cross_entropy_with_logits(z,m,pos_weight=mw)+dice.mean()
    for name,relations in [('geometry_without_neighbors',False),('geometry_with_neighbors',True)]:
        torch.manual_seed(431);model=GeometryModel(relations).cuda();rng=np.random.default_rng(431);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];best=1e9
        for step in range(1,1801):
            model.train();x,m,y,g,l=data['train'];ids=rng.integers(len(x),size=12);a,z=model(x[ids]);aux=local_loss(model,g[ids],l[ids]);loss=global_loss(a,z,y[ids],m[ids])+aux;opt.zero_grad();loss.backward();opt.step()
            if step%100==0:
                model.eval();x,m,y,g,l=data['selection'];total=0
                with torch.no_grad():
                    for i in range(0,len(x),12):a,z=model(x[i:i+12]);total+=float(global_loss(a,z,y[i:i+12],m[i:i+12]))*len(a)
                score=total/len(x);history.append(dict(step=step,selection=score,train=float(loss.detach()),geometry=float(aux.detach())));base.write(OUT/(name+'_training.json'),history)
                if score<best:best=score;torch.save(dict(state_dict=model.state_dict(),step=step),OUT/(name+'.pt'))
                print(name,step,score,flush=True)
        del model,opt
    del data
    evaluate(fonts);base.write(OUT/'status.json',dict(state='complete',seconds=time.time()-start,promoted=False));print('COMPLETE',flush=True)

def evaluate(fonts):
    data={}
    for part,seed,bank,recipe in [('fresh_same',71000000,fonts[:6],0),('fresh_font',72000000,fonts[7:],0),('fresh_recipe',73000000,fonts[:6],1)]:
        items=[base.generate(bank,seed+i,recipe) for i in range(420)];m=np.stack([a[1] for a in items]);g,l=geometry_targets(m)
        data[part]=(torch.tensor(np.stack([a[0] for a in items]).transpose(0,3,1,2)/255,dtype=torch.float32),torch.tensor(np.stack([a[2] for a in items])),torch.tensor(g),torch.tensor(l));base.write(OUT/(part+'_records.json'),[a[4] for a in items])
    results={}
    for name in ('original03','original04','geometry_without_neighbors','geometry_with_neighbors'):
        geometry=name.startswith('geometry');model=(GeometryModel(name=='geometry_with_neighbors') if geometry else base.Multi(True) if name=='original03' else old.Specialists(True)).cuda().eval()
        path=OUT/(name+'.pt') if geometry else HERE/'v2_tier1_03/multiview.pt' if name=='original03' else HERE/'v2_tier1_04/grid_specialists.pt'
        saved=torch.load(path,weights_only=True);model.load_state_dict(saved['state_dict']);r={'step':saved['step']}
        for part,(x,y,g,l) in data.items():
            preds=[];geos=[];links=[]
            with torch.no_grad():
                for i in range(0,len(x),12):
                    a,z=model(x[i:i+12].cuda());preds.append(a.sigmoid().cpu())
                    if geometry:geos.append(model.last_geometry.cpu());links.append(model.last_links.sigmoid().cpu())
            r[part]=old.metrics(torch.cat(preds),y)
            if geometry:
                p=torch.cat(geos);link=torch.cat(links);v=(g[:,:,0]>0)&(g[:,:,5]>=.6);angles=.5*torch.acos((p[:,:,3:5]*g[:,:,3:5]).sum(2).clamp(-1,1))*180/np.pi
                r[part]['geometry']=dict(axial_mae_degrees=float(angles[v].mean()),valid_direction_count=int(v.sum()),stroke=old.metrics(p[:,:,0].flatten()[:,None],g[:,:,0].flatten()[:,None]),connection=old.metrics(link.flatten()[:,None],l.flatten()[:,None]))
        results[name]=r;base.write(OUT/'results.json',results);print('EVAL',name,r['fresh_same']['f1'],flush=True)
        if name=='geometry_with_neighbors':
            real={}
            from PIL import Image
            with torch.no_grad():
                for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
                    profile=json.loads((base.RELEASE/'profiles'/f'{family}.json').read_text());rows=[]
                    for entry in profile['entries'][:4]:
                        im=Image.open(entry['reference_path_provenance']).convert('RGB').resize((128,64));x=torch.tensor(np.asarray(im).transpose(2,0,1)[None]/255,dtype=torch.float32,device='cuda');a,z=model(x)
                        rows.append(dict(reference=entry['reference_path_provenance'],features=a.sigmoid()[0].cpu().tolist(),geometry=model.last_geometry[0].cpu().tolist(),connections=model.last_links.sigmoid()[0].cpu().tolist(),existing_cv_profile=entry))
                    real[family]=rows
            base.write(OUT/'real_inference.json',real)
        del model

if __name__=='__main__':main()
