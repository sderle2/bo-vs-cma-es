"""Validate completed experiments and generate report tables/figures."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LABELS={'bo':'BO (learned kernel)','bo_fixed':'BO (fixed kernel)','cma':'CMA-ES','random':'Random search'}
COLORS={'bo':'#0072B2','bo_fixed':'#CC79A7','cma':'#D55E00','random':'#777777'}

def load(label):
    folder=ROOT/'results'/label
    cfg=json.loads((folder/'manifest.json').read_text())['config']
    frames=[]
    for f in cfg['functions']:
        for d in cfg['dimensions']:
            for t in cfg['trials']:
                initial=[]
                for method in cfg['methods']:
                    p=folder/f'{f}_d{d}_t{t}_{method}.csv'
                    df=pd.read_csv(p)
                    # Final timing includes the last CMA update and result assembly.
                    df.loc[df.index[-1],'elapsed']=json.loads(p.with_suffix('.json').read_text())['seconds']
                    assert len(df)==cfg['budget'],p
                    assert np.array_equal(df.evaluation,np.arange(1,cfg['budget']+1)),p
                    assert np.isfinite(df[['value','best','elapsed']]).all().all(),p
                    assert np.all(np.diff(df.best)<=1e-12),p
                    np.testing.assert_allclose(df.best,np.minimum.accumulate(df.value))
                    assert df.value.min()>=-1e-9,p
                    xs=np.array([json.loads(x) for x in df.x])
                    assert xs.shape==(cfg['budget'],d)
                    assert np.all((xs>=0)&(xs<=1))
                    initial.append(xs[:cfg['n_initial']])
                    frames.append(df)
                for x in initial[1:]:np.testing.assert_array_equal(x,initial[0])
    return pd.concat(frames,ignore_index=True)

def main():
    data=load('main');dest=ROOT/'results'/'summaries';dest.mkdir(exist_ok=True)
    figdir=ROOT/'report'/'figures';figdir.mkdir(exist_ok=True)
    checkpoints=data[data.evaluation.isin([50,100,200])]
    summary=checkpoints.groupby(['function','dimension','evaluation','method']).agg(
        median_gap=('best','median'),q25=('best',lambda x:x.quantile(.25)),
        q75=('best',lambda x:x.quantile(.75)),median_seconds=('elapsed','median')).reset_index()
    summary.to_csv(dest/'checkpoints.csv',index=False)
    finals=data[data.evaluation==200].copy();finals.to_csv(dest/'final_runs.csv',index=False)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(11,9))
    for i,f in enumerate(['sphere','ellipsoid','rastrigin']):
        for j,d in enumerate([2,5,10]):
            ax=axes[i,j]
            for method in ['bo','cma','random']:
                sub=data[(data.function==f)&(data.dimension==d)&(data.method==method)]
                pivot=sub.pivot(index='evaluation',columns='trial',values='best').loc[10:]
                q=np.quantile(pivot,[.25,.5,.75],axis=1)
                ax.plot(pivot.index,np.maximum(q[1],1e-10),color=COLORS[method],label=LABELS[method])
                ax.fill_between(pivot.index,np.maximum(q[0],1e-10),np.maximum(q[2],1e-10),color=COLORS[method],alpha=.15)
            ax.set_yscale('log');ax.set_title(f'{f.title()} | d = {d}')
            if i==2:ax.set_xlabel('Total objective evaluations')
            if j==0:ax.set_ylabel('Best objective gap (log scale)')
            ax.grid(alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(figdir/'main_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,f in zip(axes,['sphere','ellipsoid','rastrigin']):
        for m in ['bo','cma','random']:
            sub=finals[(finals.function==f)&(finals.method==m)]
            g=sub.groupby('dimension').elapsed.median()
            ax.plot(g.index,g.values,'o-',label=LABELS[m],color=COLORS[m])
        ax.set(title=f.title(),xlabel='Dimension',yscale='log',xticks=[2,5,10]);ax.grid(alpha=.18)
    axes[0].set_ylabel('Median seconds / 200 evaluations')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'timing.png',dpi=180);plt.close(fig)
    if '--main-only' not in sys.argv and (ROOT/'results'/'followup'/'manifest.json').exists():
        follow=load('followup');end=follow[follow.evaluation==200]
        fs=end.groupby(['function','dimension','method']).agg(median_gap=('best','median'),median_seconds=('elapsed','median')).reset_index()
        fs.to_csv(dest/'followup.csv',index=False)
        paired=end.pivot(index=['function','dimension','trial'],columns='method',values='best')
        paired.to_csv(dest/'followup_pairs.csv')
        fig,axes=plt.subplots(1,len(fs.function.unique()),figsize=(10,3.8),squeeze=False)
        for ax,f in zip(axes[0],fs.function.unique()):
            for m in ['bo','bo_fixed']:
                sub=follow[(follow.function==f)&(follow.method==m)]
                pivot=sub.pivot(index='evaluation',columns='trial',values='best').loc[10:]
                q=np.quantile(pivot,[.25,.5,.75],axis=1)
                ax.plot(pivot.index,np.maximum(q[1],1e-10),label=LABELS[m],color=COLORS[m])
                ax.fill_between(pivot.index,np.maximum(q[0],1e-10),np.maximum(q[2],1e-10),alpha=.15,color=COLORS[m])
            ax.set(title=f'{f.title()} | d = 10',xlabel='Total objective evaluations',ylabel='Best gap',yscale='log');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(figdir/'followup.png',dpi=180);plt.close(fig)
    (dest/'validation.json').write_text(json.dumps({'main_runs':len(finals),'main_evaluations':len(data),'validated':True},indent=2))
    print(summary[summary.evaluation==200].to_string(index=False))

if __name__=='__main__':main()
