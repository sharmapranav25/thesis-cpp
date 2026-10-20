"""Small image-only CRNN with CTC; no pretrained or language models."""
from setup4 import CHARSET
import torch
from torch import nn
from torch.nn import functional as F

class Solver(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[];a=3
        for b,stride in [(32,2),(64,2),(128,(2,1)),(128,(2,1))]:
            layers.extend([nn.Conv2d(a,b,3,stride,1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU()]);a=b
        self.cnn=nn.Sequential(*layers);self.proj=nn.Linear(128*3,128);self.rnn=nn.LSTM(128,128,2,batch_first=True,bidirectional=True,dropout=.1);self.out=nn.Linear(256,len(CHARSET)+1)
    def forward(self,x):
        f=self.cnn(x);x=f.permute(0,3,1,2).flatten(2);x=self.proj(x);x,_=self.rnn(x);return self.out(x)

def decode(logits):
    result=[]
    for row in logits.argmax(-1).tolist():
        prev=0;chars=[]
        for value in row:
            if value and value!=prev:chars.append(CHARSET[value-1])
            prev=value
        result.append(''.join(chars))
    return result

def edit(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def metrics(labels,pred):
    return dict(images=len(labels),exact=sum(a==b for a,b in zip(labels,pred))/len(labels),cer=sum(edit(a,b) for a,b in zip(labels,pred))/sum(map(len,labels)))

