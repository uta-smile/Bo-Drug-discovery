"""Cross auxiliary response correspondence with response-informed GP fitting.

python scripts/crossed_auxiliary.py --stage smoke --repeats 2 --workers 4
python scripts/crossed_auxiliary.py --stage full --repeats 32 --workers 4

Each checkpoint is one target/repeat at the larger pre-existing budget; its
prefix also scores the smaller budget. Both budgets share history and initial
observations. Replays evaluate all policies on ST and MM trajectories; future
primary outcomes are read only by the retrospective scoring functions.
"""
from __future__ import annotations
import os
# Avoid nested BLAS oversubscription when several families run concurrently.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from molbo.surrogate import CoregionalisedGP,TanimotoGP,fingerprint_matrix,tanimoto_gram
from molbo.chem import FINGERPRINT_VERSION
from molbo.acquisition import acquisition_scores
from dense_block_campaign import FAMILIES,draw_history,shuffle_auxiliary_responses
from dense_block_h1 import build_block,drop_censored

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"crossed_auxiliary_v1"
BUDGETS={"jak":(200,300),"cyp":(74,111),"dpp":(66,99),
         "hdac":(83,125),"mmp":(55,82),"pi3k":(130,195)}
ARMS=("stl","MM","MP","PM","PP")
VERSION="crossed-history-v1"


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def code_hash():
    h=hashlib.sha256()
    for f in ("scripts/crossed_auxiliary.py","src/molbo/surrogate.py",
              "scripts/dense_block_campaign.py","src/molbo/acquisition.py"):
        h.update((ROOT/f).read_bytes())
    return h.hexdigest()


def rank_corr(a,b):
    if np.std(a)<1e-15 or np.std(b)<1e-15:return None
    return float(spearmanr(a,b).statistic)


def state_posteriors(K,obs,observed_y,aux_idx,measured,permuted,verify=False):
    """Fit using observed primary responses only; never takes the full truth."""
    cand=np.setdiff1d(np.arange(len(K)),obs)
    idx=[obs]+aux_idx
    responses={"M":[observed_y]+measured,"P":[observed_y]+permuted}
    stl=TanimotoGP().fit_gram(K[np.ix_(obs,obs)],observed_y)
    mu,sd=stl.predict_gram(K[np.ix_(cand,obs)],return_std=True)
    output={"stl":(mu,sd)}
    parameters={}
    max_error=0.
    for source in ("M","P"):
        donor=CoregionalisedGP().fit_gram(K,idx,responses[source])
        dm,ds=donor.predict_gram(K,cand,True)
        parameters[source]=donor.parameter_snapshot()
        for response in ("M","P"):
            name=source+response
            if source==response:
                output[name]=(dm,ds)
            else:
                locked=donor.recondition(responses[response])
                # Reuse identical covariance: only mean coefficients change.
                lm=locked.predict_gram(K,cand,False)
                output[name]=(lm,ds)
            if verify:
                locked=donor.recondition(responses[response])
                lm,ls=locked.predict_gram(K,cand,True)
                assert locked.parameter_snapshot()==parameters[source]
                assert np.array_equal(ls,ds),"locked variance changed"
                if source==response:
                    err=float(np.max(np.abs(lm-dm)))
                    max_error=max(max_error,err)
                    np.testing.assert_allclose(lm,dm,rtol=1e-11,atol=1e-11)
                    before=acquisition_scores("ei",dm,ds,best=float(max(observed_y)),rng=np.random.default_rng(0))
                    after=acquisition_scores("ei",lm,ls,best=float(max(observed_y)),rng=np.random.default_rng(0))
                    np.testing.assert_allclose(before,after,rtol=1e-11,atol=1e-11)
                    np.testing.assert_array_equal(np.argsort(-(before+1e-12*np.arange(len(before)))),
                                                   np.argsort(-(after+1e-12*np.arange(len(after)))))
    best=float(max(observed_y))
    scores={a:acquisition_scores("ei",*output[a],best=best,
                                 rng=np.random.default_rng(0)) for a in ARMS}
    return {"cand":cand,"posteriors":output,"scores":scores,
            "parameters":parameters,"max_same_condition_mean_error":max_error}


def replay_metrics(state,jitter,y,top,historical,take):
    """Retrospective scoring, called only after posteriors and EI are fixed."""
    cand=state["cand"]
    picks={a:cand[np.argsort(-(state["scores"][a]+jitter))[:take]] for a in ARMS}
    baseline=set(picks["stl"].tolist())
    coverage=float(np.isin(cand,historical).mean())
    summaries={}
    for arm in ARMS:
        mu,sd=state["posteriors"][arm]
        summaries[arm]={
            "selected":picks[arm].tolist(),
            "next_batch_top10":int(np.isin(picks[arm],top).sum()),
            "mean_selected_activity":float(y[picks[arm]].mean()),
            "predictive_rank_correlation":rank_corr(mu,y[cand]),
            "acquisition_rank_vs_stl":rank_corr(state["scores"][arm],state["scores"]["stl"]),
            "batch_overlap_vs_stl":len(set(picks[arm].tolist()) & baseline)/take,
            "mean_sd":float(sd.mean()),
            "history_selection_ratio":float(np.isin(picks[arm],historical).mean()/coverage) if coverage else None}
    return summaries


def run_case(K,vals,target_index,targets,repeat,seed,budgets,verify=False):
    ti=target_index
    y=vals[:,ti]
    others=[j for j in range(len(targets)) if j!=ti]
    run_seed=seed+1009*repeat+97*ti
    # This duplicates the legacy draw streams, without consuming acquisition RNG.
    hist=draw_history(K,y,150,np.random.default_rng(run_seed),cluster_frac=0.,exclude_top=False,k=10)
    aux_idx=[hist[a*50:(a+1)*50] for a in range(3)]
    measured=[vals[idx,j] for idx,j in zip(aux_idx,others)]
    permuted,provenance=shuffle_auxiliary_responses(
        vals,aux_idx,others,np.random.default_rng(seed+7717*repeat+131*ti),"history")
    multiset_errors=[]
    for a,b in zip(measured,permuted):
        np.testing.assert_array_equal(np.sort(a),np.sort(b))
        np.testing.assert_allclose([a.mean(),a.var()],[b.mean(),b.var()],rtol=1e-12,atol=1e-12)
        multiset_errors.append(float(max(abs(a.mean()-b.mean()),abs(a.var()-b.var()))))
    rng=np.random.default_rng(run_seed)
    initial=rng.choice(len(y),size=10,replace=False)
    top=np.argsort(y)[-10:]
    historical=np.unique(np.concatenate(aux_idx))
    cache={}
    replay=[]
    sequential={}
    params={}
    invariant_max=0.
    for arm in ARMS:
        g=np.random.default_rng(run_seed)
        observed=g.choice(len(y),size=10,replace=False).tolist()
        assert np.array_equal(observed,initial)
        steps=[]
        while len(observed)<max(budgets):
            obs=np.asarray(observed,dtype=np.intp)
            key=tuple(observed)
            if key not in cache:
                cache[key]=state_posteriors(K,obs,y[obs],aux_idx,measured,permuted,verify)
            state=cache[key]
            invariant_max=max(invariant_max,state["max_same_condition_mean_error"])
            params[digest(obs)]={"n_observed":len(obs),"observed":obs.tolist(),
                                 "donors":state["parameters"]}
            jitter=1e-12*g.standard_normal(len(state["cand"]))
            take=min(10,max(budgets)-len(observed))
            order=np.argsort(-(state["scores"][arm]+jitter))
            picked=state["cand"][order[:take]]
            assert len(set(picked.tolist()))==take
            assert not set(picked.tolist()) & set(observed)
            assert np.array_equal(state["cand"],np.setdiff1d(np.arange(len(y)),obs))
            if arm in ("stl","MM"):
                # Candidate truth is never passed to state_posteriors.
                replay.append({"anchor":arm,"n_observed":len(obs),"observed_hash":digest(obs),
                               "by_budget":{str(b):replay_metrics(state,jitter,y,top,historical,
                                  min(10,b-len(obs))) for b in budgets if b>len(obs)}})
            steps.append({"n_observed":len(obs),"parameter_state":digest(obs),"selected":picked.tolist()})
            observed.extend(picked.tolist())
        assert len(observed)==max(budgets) and len(set(observed))==max(budgets)
        sequential[arm]={"selected":observed,"steps":steps,
                         "recall":{str(b):float(np.isin(observed[:b],top).sum()/10) for b in budgets}}
    return {"version":VERSION,"target":targets[ti],"target_index":ti,"repeat":repeat,
            "seed":seed,"budgets":list(budgets),"n_pool":len(y),"n_init":10,"batch":10,
            "history_per_target":50,"permutation_scope":"history","xi":.01,
            "history": [i.tolist() for i in aux_idx],"history_hash":digest(hist),
            "initial":initial.tolist(),"initial_hash":digest(initial),
            "measured_response_hash":digest(np.concatenate(measured)),
            "permuted_response_hash":digest(np.concatenate(permuted)),
            "permutations":provenance,"sequential":sequential,"replay":replay,
            "parameter_states":params,
            "audit":{"clean":True,"max_multiset_moment_error":max(multiset_errors),
                     "max_same_condition_mean_error":invariant_max,
                     "hidden_outcomes_used_for_fitting":False}}


def run_family(family,stage,repeats,seed):
    started=time.time()
    dest=OUT/stage/family
    dest.mkdir(parents=True,exist_ok=True)
    mapping=FAMILIES[family]
    block,path=build_block(mapping)
    targets=[t for t in mapping.values() if t in block.columns]
    block,_=drop_censored(block,targets,pile_frac=.02,verbose=False)
    X=fingerprint_matrix(block.smiles.tolist())
    K=tanimoto_gram(X,X)
    vals=block[targets].to_numpy(float)
    reference=json.loads((ROOT/"results"/f"winleak_{family}_b{max(BUDGETS[family])}.json").read_text())
    assert len(block)==reference["n_pool"]
    signature={"version":VERSION,"code_hash":code_hash(),"fingerprint_version":FINGERPRINT_VERSION,
               "block_hash":hashlib.sha256(path.read_bytes()).hexdigest(),"pool_hash":digest(vals),
               "kernel_hash":digest(K),"seed":seed,"family":family,"stage":stage}
    elapsed=[]
    for ti,target in enumerate(targets):
        for repeat in range(repeats):
            outfile=dest/f"{target}_r{repeat:02d}.json"
            if outfile.exists():
                prior=json.loads(outfile.read_text())
                if prior.get("signature")!=signature:
                    raise RuntimeError(f"checkpoint provenance mismatch: {outfile}")
                assert prior["audit"]["clean"]
                continue
            tick=time.time()
            case=run_case(K,vals,ti,targets,repeat,seed,BUDGETS[family],verify=stage=="smoke")
            case["signature"]=signature
            case["seconds"]=time.time()-tick
            temp=outfile.with_suffix(".tmp")
            temp.write_text(json.dumps(case,allow_nan=False),encoding="utf8")
            temp.replace(outfile)
            elapsed.append(case["seconds"])
            print(f"{stage} {family} {target} repeat {repeat+1}/{repeats}: {case['seconds']:.1f}s",flush=True)
    return {"family":family,"repeats":repeats,"seconds":time.time()-started,
            "new_cases":len(elapsed),"mean_case_seconds":float(np.mean(elapsed)) if elapsed else 0.}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["smoke","full"],required=True)
    p.add_argument("--repeats",type=int,required=True)
    p.add_argument("--families",nargs="+",choices=list(BUDGETS),default=list(BUDGETS))
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=0)
    args=p.parse_args()
    if args.stage=="smoke" and args.repeats!=2:p.error("smoke requires two repeats")
    if args.stage=="full" and args.repeats!=32:p.error("full requires 32 repeats")
    os.chdir(ROOT)
    if args.stage=="full":
        for family in args.families:
            for ti in FAMILIES[family].values():
                for r in range(2):
                    check=OUT/"smoke"/family/f"{ti}_r{r:02d}.json"
                    o=json.loads(check.read_text())
                    assert o["audit"]["clean"] and o["signature"]["code_hash"]==code_hash()
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_family,f,args.stage,args.repeats,args.seed) for f in args.families]
        for future in as_completed(futures):results.append(future.result())
    (OUT/args.stage/"runtime.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    print(json.dumps(results,indent=2),flush=True)

if __name__=="__main__":main()
