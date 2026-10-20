from setup4 import ROOT,U,CHARSET
import torch
from solver4 import Solver,decode,edit
from corpus4 import render_job,C

def main():
    torch.set_num_threads(2);model=Solver();x=torch.rand(2,3,48,192);y=model(x)
    assert y.shape==(2,48,63) and torch.isfinite(y).all()
    labels=torch.tensor([1,1,2,3]);loss=torch.nn.functional.ctc_loss(y.log_softmax(-1).transpose(0,1),labels,torch.tensor([48,48]),torch.tensor([2,2]),zero_infinity=False)
    loss.backward();assert torch.isfinite(loss)
    logits=torch.full((1,5,63),-10.);ids=[1,1,0,1,2]
    for i,v in enumerate(ids):logits[0,i,v]=10
    assert decode(logits)==[CHARSET[0]+CHARSET[0]+CHARSET[1]]
    assert edit('abc','axc')==1 and edit('abc','')==3
    U.dump(ROOT/'checks.json',dict(passed=True,ctc_forward_backward_finite=True,decoder_repeats_and_blank=True,edit_distance=True,parameters=sum(p.numel() for p in model.parameters())))
    print('Solver checks passed',flush=True)

if __name__=='__main__':main()
