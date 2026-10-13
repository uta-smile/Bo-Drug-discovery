"""Independently reconstruct crossed-control outcomes before emitting paper results."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import t,wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_block_campaign import FAMILIES
from dense_block_h1 import build_block,drop_censored
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results"/"crossed_auxiliary_v1"
BUDGETS={"jak":(200,300),"cyp":(74,111),"dpp":(66,99),"hdac":(83,125),"mmp":(55,82),"pi3k":(130,195)}
ARMS=("stl","MM","MP","PM","PP")
CONTRASTS={"primary":("MM","PP"),"measured_gain":("MM","stl"),"permuted_gain":("PP","stl"),
 "response_measured_fit":("MM","MP"),"response_permuted_fit":("PM","PP"),
 "fit_measured_response":("MM","PM"),"fit_permuted_response":("MP","PP")}
METRICS=("next_batch_top10","mean_selected_activity","predictive_rank_correlation",
 "acquisition_rank_vs_stl","batch_overlap_vs_stl","mean_sd","history_selection_ratio")

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def estimate(values,confirmatory=False):
 v=np.array(values,dtype=float);sem=float(v.std(ddof=1)/np.sqrt(len(v)));half=float(t.ppf(.975,len(v)-1)*sem)
 out={"mean":float(v.mean()),"sem":sem,"ci95":[float(v.mean()-half),float(v.mean()+half)],
  "families":len(v),"positive":int((v>0).sum()),"values":v.tolist(),"confirmatory":confirmatory}
 if confirmatory:
  out["p_two_sided"]=float(wilcoxon(v,alternative="two-sided",method="auto").pvalue) if np.any(v) else 1.
  out["minimum_two_sided_p"]=.03125
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=["smoke","full"],required=True)
 p.add_argument("--emit-tex",action="store_true");a=p.parse_args();repeats=2 if a.stage=="smoke" else 32
 if a.emit_tex and a.stage!="full":p.error("only full results can update the manuscript")
 families={};records=[];hashes=set();max_legacy_error=0.;max_moment_error=0.
 for family,budgets in BUDGETS.items():
  block,_=build_block(FAMILIES[family]);targets=list(FAMILIES[family].values())
  block,_=drop_censored(block,targets,pile_frac=.02,verbose=False);vals=block[targets].to_numpy(float)
  files=sorted((BASE/a.stage/family).glob("*.json"));assert len(files)==4*repeats,(family,len(files))
  outcomes={str(b):{arm:[] for arm in ARMS} for b in budgets}
  replays={anchor:{arm:{m:[] for m in METRICS} for arm in ARMS} for anchor in ("stl","MM")}
  seen=set();legacy={str(b):json.loads((ROOT/"results"/f"winleak_{family}_b{b}.json").read_text()) for b in budgets}
  for file in files:
   c=json.loads(file.read_text());ti=c["target_index"];r=c["repeat"]
   assert (ti,r) not in seen and 0<=ti<4 and 0<=r<repeats;seen.add((ti,r))
   assert c["target"]==targets[ti] and c["budgets"]==list(budgets)
   assert c["signature"]["pool_hash"]==sha(vals)
   assert c["permutation_scope"]=="history" and c["audit"]["clean"];hashes.add(c["signature"]["code_hash"])
   hist=np.concatenate(c["history"]).astype(np.intp)
   assert sha(hist)==c["history_hash"] and len(hist)==150 and len(set(hist.tolist()))==150
   initial=np.array(c["initial"],dtype=np.intp);assert len(initial)==10 and sha(initial)==c["initial_hash"]
   seed=c["seed"]+1009*r+97*ti
   np.testing.assert_array_equal(hist,np.random.default_rng(seed).choice(np.arange(len(vals)),150,replace=False))
   np.testing.assert_array_equal(initial,np.random.default_rng(seed).choice(len(vals),10,replace=False))
   measured=[];permuted=[]
   for ai,rec in enumerate(c["permutations"]):
    loc=np.array(rec["locations"],dtype=np.intp);src=np.array(rec["source_indices"],dtype=np.intp);j=rec["target_column"]
    assert np.array_equal(loc,np.asarray(c["history"][ai]))
    assert j==[j for j in range(4) if j!=ti][ai]
    assert j!=ti and len(loc)==50 and np.array_equal(np.sort(loc),np.sort(src))
    m=vals[loc,j];q=vals[src,j];assert np.array_equal(np.sort(m),np.sort(q))
    max_moment_error=max(max_moment_error,abs(float(m.mean()-q.mean())),abs(float(m.var()-q.var())))
    measured.append(m);permuted.append(q)
   assert sha(np.concatenate(measured))==c["measured_response_hash"]
   assert sha(np.concatenate(permuted))==c["permuted_response_hash"]
   y=vals[:,ti];top=set(np.argsort(y)[-10:].tolist())
   for arm in ARMS:
    run=c["sequential"][arm];idx=run["selected"]
    assert len(idx)==max(budgets) and len(set(idx))==max(budgets) and idx[:10]==initial.tolist()
    for step in run["steps"]:
     state=c["parameter_states"][step["parameter_state"]]
     observed=np.array(idx[:step["n_observed"]],dtype=np.intp)
     np.testing.assert_array_equal(observed,state["observed"])
     assert sha(observed)==step["parameter_state"]
     for donor in state["donors"].values():
      np.testing.assert_allclose(donor["task_means"][0],y[observed].mean(),rtol=1e-12,atol=1e-12)
      scale=float(y[observed].std());scale=scale if scale>1e-12 else 1.
      np.testing.assert_allclose(donor["task_scales"][0],scale,rtol=1e-12,atol=1e-12)
    for b in budgets:
     value=len(set(idx[:b])&top)/10;assert value==run["recall"][str(b)];outcomes[str(b)][arm].append(value)
     if arm in ("stl","MM"):
      old=legacy[str(b)]["arms"]["stl" if arm=="stl" else "mt"]["recall"][ti*32+r]
      max_legacy_error=max(max_legacy_error,abs(value-old))
   for anchor in ("stl","MM"):
    for arm in ARMS:
     for metric in METRICS:
      budget_means=[]
      for b in budgets:
       rows=[v["by_budget"][str(b)][arm] for v in c["replay"] if v["anchor"]==anchor and str(b) in v["by_budget"]]
       for row in rows:assert row["next_batch_top10"]==len(set(row["selected"])&top)
       values=[row[metric] for row in rows if row[metric] is not None]
       if values:budget_means.append(float(np.mean(values)))
      if budget_means:replays[anchor][arm][metric].append(float(np.mean(budget_means)))
   records.append({"family":family,"target":targets[ti],"repeat":r,"path":str(file.relative_to(ROOT)),
    "sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
  gains={arm:float(np.mean([np.mean(np.array(outcomes[str(b)][arm])-np.array(outcomes[str(b)]["stl"]))*100 for b in budgets])) for arm in ARMS}
  contrasts={name:float(np.mean([np.mean(np.array(outcomes[str(b)][x])-np.array(outcomes[str(b)][z]))*100 for b in budgets])) for name,(x,z) in CONTRASTS.items()}
  contrasts["interaction"]=gains["MM"]-gains["MP"]-gains["PM"]+gains["PP"]
  replay={anchor:{arm:{m:float(np.mean(v)) if v else None for m,v in data.items()} for arm,data in arms.items()} for anchor,arms in replays.items()}
  families[family]={"gains":gains,"contrasts":contrasts,"replay":replay,
   "budget_means":{b:{arm:float(np.mean(v)) for arm,v in arms.items()} for b,arms in outcomes.items()}}
 assert len(hashes)==1 and max_legacy_error==0.,("baseline mismatch",max_legacy_error)
 summaries={name:estimate([f["contrasts"][name] for f in families.values()],name=="primary" and a.stage=="full") for name in list(CONTRASTS)+["interaction"]}
 replay_summary={anchor:{arm:{m:float(np.mean([f["replay"][anchor][arm][m] for f in families.values() if f["replay"][anchor][arm][m] is not None])) for m in METRICS} for arm in ARMS} for anchor in ("stl","MM")}
 report={"stage":a.stage,"families":families,"estimates":summaries,"replay":replay_summary,
  "audit":{"clean":True,"cases":len(records),"budget_campaigns_per_arm":len(records)*2,
   "max_legacy_recall_difference":max_legacy_error,"max_moment_error":max_moment_error,
   "code_hash":list(hashes)[0],"statistical_unit":"protein family"},"manifest":records}
 (BASE/a.stage/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
 if a.emit_tex:
  out=ROOT/"report"/"generated"
  def macro(name,value):return "\\newcommand{\\cross"+name+"}{"+value+"}\n"
  tex="% Generated by scripts/summarize_crossed_auxiliary.py; do not edit.\n"
  for name,rec in summaries.items():
   suffix="".join(w.capitalize() for w in name.split("_"));tex+=macro(suffix,f"{rec['mean']:+.2f}")
   tex+=macro(suffix+"Ci",f"[{rec['ci95'][0]:+.2f},{rec['ci95'][1]:+.2f}]")
   tex+=macro(suffix+"Positive",str(rec["positive"]))
  tex+=macro("PrimaryP",f"{summaries['primary']['p_two_sided']:.5f}");tex+=macro("Campaigns",str(repeats*4))
  for anchor in ("stl","MM"):
   for arm in ARMS:
    prefix=("StlAnchor" if anchor=="stl" else "MtAnchor")+arm.upper()
    tex+=macro(prefix+"BatchHits",f"{replay_summary[anchor][arm]['next_batch_top10']:.3f}")
  (out/"macros_crossed.tex").write_text(tex,encoding="utf8")
  rows=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Family & $\mathrm{MM}$ $-$ \stl{} & $\mathrm{MP}$ $-$ \stl{} & $\mathrm{PM}$ $-$ \stl{} & $\mathrm{PP}$ $-$ \stl{} \\",r"\midrule"]
  for family in BUDGETS:
   label="PI3K" if family=="pi3k" else family.upper()
   values=["$"+f"{families[family]['gains'][arm]:+.2f}"+"$" for arm in ARMS[1:]]
   rows.append(label+" & "+" & ".join(values)+r" \\")
  means=["$"+f"{np.mean([f['gains'][arm] for f in families.values()]):+.2f}"+"$" for arm in ARMS[1:]]
  rows += [r"\midrule","Mean & "+" & ".join(means)+r" \\",r"\bottomrule",r"\end{tabular}"]
  (out/"table_crossed.tex").write_text("\n".join(rows)+"\n",encoding="utf8")
 print(json.dumps({"audit":report["audit"],"estimates":summaries,"replay":replay_summary},indent=2))

if __name__=="__main__":main()
