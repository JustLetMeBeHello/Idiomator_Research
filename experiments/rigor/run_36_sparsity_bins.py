"""run_36_sparsity_bins.py: Paper B 6.19 (CPU). Strip-mode QA-BIO gap of byte-BPE encoders (no punct artifact) binned by span sparsity
(span words / sentence words) and by span length, per dataset. Tests whether the residual QA edge tracks target sparsity.
Run: .venv/bin/python3 experiments/rigor/run_36_sparsity_bins.py   (reads the Drive mount; prints only)
"""
import os,sys,glob,re
os.environ["BBPE"]="1"
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, run_08_extended_gold as r
D=glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive"))[0]
rng=np.random.default_rng(0)
def feats(row):
    t=row["sentence"]; nw=len(t.split()); sw=len(t[row["span_start"]:row["span_end"]].split())
    return sw/max(nw,1), sw, nw
def collect(root,encs):
    out=[]  # (frac, span_words, sent_words, strip_gap)
    for enc in encs:
        for s in (42,123,7):
            qp,bp=r.pred_path(root,enc,"joint",s),r.pred_path(root,enc,"bio",s)
            if not(qp.exists() and bp.exists()): continue
            q={r.row_key(x):x for x in r.load_rows(qp)}; b={r.row_key(x):x for x in r.load_rows(bp)}
            for k in q.keys()&b.keys():
                f=feats(q[k]); out.append((*f, r.score_row(q[k],"strip")[0]-r.score_row(b[k],"strip")[0]))
    return np.array(out)
def ci(x):
    bs=x[rng.integers(0,len(x),(5000,len(x)))].mean(1); return np.quantile(bs,[.025,.975])
def bins(a,col,edges,label):
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(a[:,col]>=lo)&(a[:,col]<hi)
        if m.sum()<30: continue
        g=a[m,3]; c=ci(g); print(f"   {label}[{lo:.2f},{hi:.2f}) n={m.sum():5d} gap {g.mean():+.3f} [{c[0]:+.3f},{c[1]:+.3f}]")
roots={"WikiANN EN":(f"{D}/IdiomatorRigor_NER_en_bbpe",["roberta","modernbert"]),
 "ABSA rest":(f"{D}/IdiomatorRigor_NER_absa_restaurants",["roberta","modernbert"]),
 "ABSA lap":(f"{D}/IdiomatorRigor_NER_absa_laptops",["roberta","modernbert"]),
 "ABSA rest ow1":(f"{D}/IdiomatorRigor_NER_absa_restaurants_ow1",["roberta","modernbert"]),
 "ABSA lap ow1":(f"{D}/IdiomatorRigor_NER_absa_laptops_ow1",["roberta","modernbert"])}
A={}
for n,(p,e) in roots.items():
    a=collect(p,e); A[n]=a
    print(f"\n{n}: n={len(a)} mean span_frac {a[:,0].mean():.3f} median {np.median(a[:,0]):.3f} mean span_words {a[:,1].mean():.2f} mean sent_words {a[:,2].mean():.1f} strip gap {a[:,3].mean():+.3f}")
    bins(a,0,[0,.06,.1,.15,.25,1.01],"frac")
    bins(a,1,[1,2,3,4,100],"spanw")
