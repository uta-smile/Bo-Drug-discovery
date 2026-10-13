"""Distribute independent crossed-control checkpoints across CPU workers.

Only orchestration differs from crossed_auxiliary.py. Scientific computations,
input construction, checkpoint schema, and code hashes use that module unchanged.
"""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import argparse,json,time,hashlib
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from crossed_auxiliary import (ROOT,OUT,VERSION,BUDGETS,FAMILIES,FINGERPRINT_VERSION,
                              code_hash,digest,run_case,build_block,drop_censored,
                              fingerprint_matrix,tanimoto_gram)

@lru_cache(maxsize=6)
def inputs(family):
    mapping=FAMILIES[family]
    block,path=build_block(mapping)
    targets=[t for t in mapping.values() if t in block.columns]
    block,_=drop_censored(block,targets,pile_frac=.02,verbose=False)
    K=tanimoto_gram(fingerprint_matrix(block.smiles.tolist()),fingerprint_matrix(block.smiles.tolist()))
    vals=block[targets].to_numpy(float)
    old=json.loads((ROOT/"results"/f"winleak_{family}_b{max(BUDGETS[family])}.json").read_text())
    assert len(block)==old["n_pool"]
    return K,vals,targets,path


def checkpoint(task):
    family,ti,repeat,seed,stage=task
    os.chdir(ROOT)
    K,vals,targets,path=inputs(family)
    dest=OUT/stage/family
    dest.mkdir(parents=True,exist_ok=True)
    outfile=dest/f"{targets[ti]}_r{repeat:02d}.json"
    signature={"version":VERSION,"code_hash":code_hash(),"fingerprint_version":FINGERPRINT_VERSION,
               "block_hash":hashlib.sha256(path.read_bytes()).hexdigest(),"pool_hash":digest(vals),
               "kernel_hash":digest(K),"seed":seed,"family":family,"stage":stage}
    if outfile.exists():
        prior=json.loads(outfile.read_text())
        if prior.get("signature")!=signature:raise RuntimeError(f"checkpoint mismatch: {outfile}")
        assert prior["audit"]["clean"]
        return family,targets[ti],repeat,0.
    start=time.time()
    case=run_case(K,vals,ti,targets,repeat,seed,BUDGETS[family],verify=stage=="smoke")
    case["signature"]=signature
    case["seconds"]=time.time()-start
    tmp=outfile.with_suffix(".tmp")
    tmp.write_text(json.dumps(case,allow_nan=False),encoding="utf8")
    tmp.replace(outfile)
    return family,targets[ti],repeat,case["seconds"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--stage",choices=["smoke","full"],default="full")
    a=p.parse_args()
    os.chdir(ROOT)
    if a.stage=="full":
        for family in BUDGETS:
            for target in FAMILIES[family].values():
                for repeat in range(2):
                    c=json.loads((OUT/"smoke"/family/f"{target}_r{repeat:02d}.json").read_text())
                    assert c["audit"]["clean"] and c["signature"]["code_hash"]==code_hash()
    # Round-robin across repeats and targets, so expensive JAK work can use
    # every available worker and small families continue to report progress.
    repeats=2 if a.stage=="smoke" else 32
    tasks=[(f,ti,r,a.seed,a.stage) for r in range(repeats) for ti in range(4) for f in BUDGETS]
    start=time.time()
    timings=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(checkpoint,t) for t in tasks]
        for done,future in enumerate(as_completed(futures),1):
            family,target,r,seconds=future.result()
            timings.append({"family":family,"target":target,"repeat":r,"seconds":seconds})
            print(f"{a.stage} {done}/{len(tasks)}: {family} {target} r{r:02d} {seconds:.1f}s",flush=True)
    (OUT/a.stage/"runtime.json").write_text(json.dumps({"seconds":time.time()-start,
        "runner_hash":hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        "cases":timings},indent=2),encoding="utf8")

if __name__=="__main__":main()
