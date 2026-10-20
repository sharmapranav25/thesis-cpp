"""Unmodified pinned upstream readers with optional original auxiliary heads."""
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import UPSTREAM, CHARSET, groups
sys.path[1:1]=[str(UPSTREAM/'parseq'),str(UPSTREAM/'OpenOCR')]

class Reader(nn.Module):
    def __init__(self, name, auxiliary, seed=731):
        super().__init__();self.name=name;self.auxiliary=auxiliary;self.feature=None
        if name in ('crnn','parseq'):
            from strhub.models.utils import create_model
            overrides=dict(charset_train=CHARSET,charset_test=CHARSET,max_label_length=12,img_size=[32,192],batch_size=128,lr=.0003,weight_decay=.01,warmup_pct=.1)
            if name=='parseq':overrides.update(embed_dim=192,enc_num_heads=3,enc_depth=6,dec_num_heads=6)
            self.net=create_model(name,pretrained=False,**overrides);self.net.log=lambda *args,**kwargs:None
            if name=='parseq':
                self.net.rng=np.random.default_rng(seed);module=self.net.model.encoder;dim=192
            else:module=self.net.model.cnn;dim=512
            module.register_forward_hook(self.capture)
        elif name=='svtr':
            from openrec.modeling.encoders.svtrnet import SVTRNet
            self.net=SVTRNet(img_size=[32,192],embed_dim=[64,128,256],depth=[3,6,3],num_heads=[2,4,8],mixer=['Local']*6+['Global']*6,out_channels=192,out_char_num=48)
            dim=192;self.ctc=nn.Linear(dim,len(CHARSET)+1);self.net.register_forward_hook(self.capture)
        else:raise ValueError(name)
        self.count=nn.Linear(dim,7);self.group=nn.Linear(dim,3)

    def capture(self,module,inputs,output):
        self.feature=output.mean((2,3)) if output.ndim==4 else output.mean(1)

    def to(self,*args,**kwargs):
        super().to(*args,**kwargs)
        if self.name in ('crnn','parseq'):self.net.to(*args,**kwargs)
        return self

    def loss(self,images,labels):
        if self.name=='svtr':
            logits=self.ctc(self.net(images));targets=torch.tensor([CHARSET.index(c)+1 for t in labels for c in t],device=images.device)
            lengths=torch.tensor(list(map(len,labels)),dtype=torch.long)
            assert all(len(t)+sum(a==b for a,b in zip(t,t[1:]))<=logits.shape[1] for t in labels)
            recognition=F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),targets,torch.full_like(lengths,logits.shape[1]),lengths,zero_infinity=False)
        else:recognition=self.net.training_step((images,labels),0)
        loss=recognition
        if self.auxiliary:
            counts=torch.tensor([len(t)-2 for t in labels],device=images.device)
            gs=torch.tensor([groups(t) for t in labels],dtype=torch.float32,device=images.device)
            loss=loss+.25*F.cross_entropy(self.count(self.feature).float(),counts)+.25*F.binary_cross_entropy_with_logits(self.group(self.feature).float(),gs)
        return loss,recognition.detach()

    @torch.no_grad()
    def predict(self,images):
        if self.name=='svtr':
            logits=self.ctc(self.net(images));prob=logits.float().softmax(-1);values=prob.argmax(-1);pred=[];conf=[]
            for row,ps in zip(values.tolist(),prob.max(-1).values.tolist()):
                last=0; chars=[];scores=[]
                for value,score in zip(row,ps):
                    if value and value!=last:chars.append(CHARSET[value-1]);scores.append(score)
                    last=value
                pred.append(''.join(chars));conf.append(float(np.exp(np.mean(np.log(np.maximum(scores,1e-12))))) if scores else 0.)
        else:
            logits=self.net(images);pred,confs=self.net.tokenizer.decode(logits.float().softmax(-1))
            conf=[float(c.clamp_min(1e-12).log().mean().exp()) if c.numel() else 0. for c in confs]
        aux={}
        if self.auxiliary:aux=dict(count_logits=self.count(self.feature).float().cpu().numpy(),group_logits=self.group(self.feature).float().cpu().numpy())
        return pred,conf,aux
