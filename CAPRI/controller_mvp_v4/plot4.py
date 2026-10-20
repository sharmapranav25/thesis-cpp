"""Static, reproducible scientific figures from the frozen experiment results."""
from setup4 import ROOT,U,FAMILIES,ARMS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    colors={'hybrid':'#64748b','single':'#d97706','diverse':'#0f766e'}
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,layout='constrained')
    for arm in ARMS:
        for seed,style in [(43103,'-'),(43109,'--')]:
            history=U.read(ROOT/'solvers'/f'{arm}_{seed}'/'history.json')
            for ax,metric in zip(axes,['exact','cer']):
                ax.plot([r['epoch'] for r in history],[100*r['validation'][metric] for r in history],style,color=colors[arm],label=arm if seed==43103 else None)
    axes[0].set(title='Common synthetic validation — all six runs',ylabel='Exact CAPTCHA accuracy (%)',ylim=(0,100))
    axes[1].set(ylabel='Character error rate (%)',xlabel='Epoch',ylim=(0,110),xticks=[1,5,10,15,20])
    axes[0].legend();axes[1].text(.02,.96,'Solid: seed 43103   Dashed: seed 43109',transform=axes[1].transAxes,va='top')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(ROOT/'learning_curves.png',dpi=160);plt.close(fig)
    results=U.read(ROOT/'evaluation/results.json')['families'];families=[f for f in FAMILIES if 'scores' in results[f]]
    if not families:return
    fig,axes=plt.subplots(2,1,figsize=(9,8),sharex=True,layout='constrained');x=np.arange(len(families));width=.24
    for j,arm in enumerate(ARMS):
        for ax,metric in zip(axes,['exact','cer']):
            pos=x+(j-1)*width;values=[100*results[f]['scores'][arm]['mean_'+metric] for f in families]
            ax.bar(pos,values,width,color=colors[arm],label=arm)
            for i,f in enumerate(families):
                ys=[100*r[metric] for r in results[f]['scores'][arm]['seeds']]
                ax.scatter([pos[i]-.035,pos[i]+.035],ys,s=15,c='black',zorder=3)
    axes[0].set(title='Held-out target recognition — frozen readers',ylabel='Exact CAPTCHA accuracy (%)',ylim=(0,100));axes[0].legend()
    axes[1].set(ylabel='Character error rate (%)')
    axes[1].set_xticks(x,[f.replace('SimpleCaptcha','Simple\nCaptcha')+'\nn='+str(results[f]['images'])+(' (reused)' if results[f]['split_status']=='reused_diagnostic' else '') for f in families])
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Bars: mean of two fixed seeds. Dots: individual seeds.\nSimpleCaptcha is primary; other pools are small or diagnostic.',fontsize=11)
    fig.savefig(ROOT/'target_scores.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
