"""Evaluate survival and uplift on real datasets with held-out observations.

Input paths and source links are documented in the survival and uplift guides.
This example is a reproducible research demonstration, not a clinical or marketing policy.
"""
from bisect import bisect_left, bisect_right
from collections import Counter
from math import sqrt
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mltreelib import ModelTree

DATA = Path("datasets/external/real_eval")
OUT = Path("docs/assets/guides")
OUT.mkdir(parents=True, exist_ok=True)


def censor_km(times, events):
    """Censoring survival G(t), estimated from training observations."""
    counts = Counter(times)
    censored = Counter(t for t, event in zip(times, events) if not event)
    at_risk = len(times)
    steps, probs, g = [], [], 1.0
    for t in sorted(counts):
        g *= 1 - censored[t] / at_risk
        steps.append(t); probs.append(g); at_risk -= counts[t]
    def g_at(t, left=False):
        i = (bisect_left if left else bisect_right)(steps, t) - 1
        return 1.0 if i < 0 else probs[i]
    return g_at


def brier(times, events, predictions, horizon, censor):
    g_h = censor(horizon)
    if g_h <= 0:
        raise ValueError("Censoring survival is zero at the horizon")
    total = 0.0
    for t, e, pred in zip(times, events, predictions):
        if t <= horizon and e:
            g = censor(t, left=True)
            if g <= 0:
                raise ValueError("Censoring survival is zero before an event")
            total += pred ** 2 / g
        elif t > horizon:
            total += (1 - pred) ** 2 / g_h
    return total / len(times)


def survival():
    df = pd.read_csv(DATA / "flchain.csv")
    features = ["age", "sex", "kappa", "lambda", "creatinine"]
    x = [[None if pd.isna(v) else v for v in row] for row in df[features].itertuples(index=False, name=None)]
    t = df["futime"].astype(float).tolist(); e = df["death"].astype(float).tolist()
    ids = list(range(len(x))); Random(2026).shuffle(ids)
    tr, te = ids[:5500], ids[5500:]
    pick = lambda a, ix: [a[i] for i in ix]
    xtr, xte = pick(x,tr),pick(x,te); ttr,tte=pick(t,tr),pick(t,te); etr,ete=pick(e,tr),pick(e,te)
    model=ModelTree(leaf_family="survival",max_depth=2,min_rows=100,prune="none")
    model.fit(xtr,(ttr,etr),col_names=features,dtypes=["numeric","categorical","numeric","numeric","numeric"])
    base=ModelTree(leaf_family="survival",max_depth=0,min_rows=100)
    base.fit(xtr,(ttr,etr),col_names=features,dtypes=["numeric","categorical","numeric","numeric","numeric"])
    grid=[365.,1095.,1825.]; pred=model.predict(xte,kind="survival",times=grid); baseline=base.predict(xte,kind="survival",times=grid); g=censor_km(ttr,etr)
    print("FLCHAIN train/test",len(tr),len(te),"test deaths/censored",int(sum(ete)),len(ete)-int(sum(ete)))
    print("FLCHAIN training missing creatinine",sum(row[4] is None for row in xtr),"test",sum(row[4] is None for row in xte))
    tree_scores=[]; base_scores=[]
    for j,h in enumerate(grid):
        a=brier(tte,ete,[v[j] for v in pred],h,g);b=brier(tte,ete,[v[j] for v in baseline],h,g)
        tree_scores.append(a);base_scores.append(b)
        print("FLCHAIN horizon",int(h),"G",round(g(h),4),"at-risk",sum(v>=h for v in tte),"tree",round(a,4),"baseline",round(b,4))
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(grid,tree_scores,marker="o",label="tree");ax.plot(grid,base_scores,marker="o",label="global KM")
    ax.set(xlabel="Days",ylabel="IPCW Brier, lower is better",title="FLCHAIN: held-out evaluation");ax.legend();fig.tight_layout();fig.savefig(OUT/"real-survival.svg");plt.close(fig)


def group_value(y,w,selected):
    a=[v for v,t,s in zip(y,w,selected) if s and t==1];b=[v for v,t,s in zip(y,w,selected) if s and t==0];n=sum(selected)
    if not n: return 0.,0,0,0,None
    if len(a)<2 or len(b)<2: raise ValueError("Selected group requires at least two observations per arm")
    diff=sum(a)/len(a)-sum(b)/len(b);q=n/len(y)
    def var(v):
        m=sum(v)/len(v);return sum((z-m)**2 for z in v)/(len(v)-1)
    se=q*sqrt(var(a)/len(a)+var(b)/len(b))
    return q*diff,n,len(a),len(b),(q*diff-1.96*se,q*diff+1.96*se)


def uplift():
    df=pd.read_csv(DATA/"hillstrom.csv")
    df=df[df["segment"].isin(["Womens E-Mail","No E-Mail"])].reset_index(drop=True)
    features=["recency","history","mens","womens","newbie","zip_code","channel"]
    types=["numeric"]*5+["categorical"]*2
    x=[[None if pd.isna(v) else v for v in row] for row in df[features].itertuples(index=False,name=None)]
    y=df["visit"].astype(float).tolist();w=[float(s=="Womens E-Mail") for s in df["segment"]]
    ids=list(range(len(x)));Random(2026).shuffle(ids); n=len(ids);tr=ids[:int(n*.6)];va=ids[int(n*.6):int(n*.8)];te=ids[int(n*.8):]
    pick=lambda a,ix:[a[i] for i in ix]
    model=ModelTree(gain="causal",leaf="mean",max_depth=2,min_rows=250,prune="none",honesty=True,seed=42)
    model.fit(pick(x,tr),(pick(y,tr),pick(w,tr)),col_names=features,dtypes=types)
    val=model.predict(pick(x,va),kind="uplift"); thresholds=sorted(set(val));thresholds=[min(val)-1]+thresholds+[max(val)+1]
    threshold=max(thresholds,key=lambda z:group_value(pick(y,va),pick(w,va),[s>z for s in val])[0])
    pred=model.predict(pick(x,te),kind="uplift");yt,wt=pick(y,te),pick(w,te)
    policies={"selected":[v>threshold for v in pred],"treat all":[True]*len(te),"treat none":[False]*len(te)}
    print("Hillstrom women vs control train/val/test",len(tr),len(va),len(te),"threshold",round(threshold,6))
    vals=[]
    for name,selected in policies.items():
        value,nsel,nt,nc,interval=group_value(yt,wt,selected);vals.append(value)
        print("Hillstrom",name,"value",round(value,5),"selected",nsel,"treated/control",nt,nc,"interval",None if interval is None else tuple(round(i,5) for i in interval))
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(list(policies),vals,color=["#00897b","#0288d1","#78909c"]);ax.axhline(0,color="black",linewidth=.8)
    ax.set(ylabel="Incremental visits per eligible customer",title="Hillstrom: randomized held-out comparison");fig.tight_layout();fig.savefig(OUT/"real-uplift.svg");plt.close(fig)


if __name__=="__main__":
    survival()
    uplift()
