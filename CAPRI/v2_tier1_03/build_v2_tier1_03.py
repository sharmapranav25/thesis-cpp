from pathlib import Path

p=Path(__file__).parent
s=(p/'run_v2_tier1_02.py').read_text()
s=s.replace("/'v2_tier1_02'", "/'v2_tier1_03'")
s=s.replace("self.context=nn.Linear(48,48)","""self.context=nn.Linear(48,48)
        # Native-resolution support bypasses the stride-8 bottleneck.
        self.detail=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.Conv2d(24,24,3,padding=1),nn.ReLU())
        self.mid=nn.Sequential(nn.Conv2d(24,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.decode=nn.Sequential(nn.Conv2d(104,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.Conv2d(48,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,5,1))
        self.presence=nn.Sequential(nn.Linear(48+10,48),nn.ReLU(),nn.Linear(48,5))""")
old="return self.presence(context),F.interpolate(self.location(spatial),size=(32,64),mode='bilinear',align_corners=False)"
new="""detail=self.detail(x);mid=F.interpolate(self.mid(detail),size=x.shape[-2:],mode='bilinear',align_corners=False);coarse=F.interpolate(spatial,size=x.shape[-2:],mode='bilinear',align_corners=False);z=self.decode(torch.cat([detail,mid,coarse],1));prob=z.sigmoid().flatten(2);summary=torch.cat([prob.mean(2),prob.topk(32,dim=2).values.mean(2)],1);return self.presence(torch.cat([context,summary],1)),z"""
assert old in s;s=s.replace(old,new)
# Same images as 02 isolate architecture; these are now development benchmarks.
s=s.replace("cv2.resize(mask.transpose(1,2,0),(64,32),interpolation=cv2.INTER_AREA).transpose(2,0,1)","mask")
s=s.replace("masks.append(mask)","masks.append(mask)")
s=s.replace("steps=1500", "steps=1800").replace("range(1,1501)","range(1,1801)")
s=s.replace("threshold=.5,limitations=", "threshold=.5,architecture='native-resolution detail + stride2 skip + stride8 context decoder; mask mean/top32 pooled presence evidence',benchmark_status='02 test/heldout reused for architectural diagnosis; development only, no fresh acceptance test',limitations=")
s=s.replace("pred=z.sigmoid()>=.5;truth=m>=.25", "pred=z.sigmoid()>=.5;truth=m>=.25")
s=s.replace("dict(f1=f1.cpu().tolist(),", "dict(precision=(tp/(tp+fp).clamp_min(1)).cpu().tolist(),recall=(tp/(tp+fn).clamp_min(1)).cpu().tolist(),f1=f1.cpu().tolist(),")
# Bound validation peak memory for native-resolution maps; batches are concatenated on CPU.
s=s.replace("aa.append(a);zz.append(z)","aa.append(a.cpu());zz.append(z.cpu())")
s=s.replace("return torch.cat(aa),torch.cat(zz),m,y", "return torch.cat(aa),torch.cat(zz),m.cpu(),y.cpu()")
s=s.replace("pos_weight=presence_weight)","pos_weight=presence_weight.to(a.device))").replace("pos_weight=mask_weight)","pos_weight=mask_weight.to(z.device))")
(p/'run_v2_tier1_03.py').write_text(s,encoding='utf8')
print('Built revision03')
