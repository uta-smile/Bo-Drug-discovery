"""Scientific invariants of crossed response/fitting controls."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from dense_block_campaign import shuffle_auxiliary_responses
from molbo.surrogate import CoregionalisedGP, TanimotoGP, tanimoto_gram
from molbo.acquisition import acquisition_scores


def panel():
    rng = np.random.default_rng(72)
    x = (rng.random((40, 64)) < .25).astype(float)
    K = tanimoto_gram(x, x)
    idx = [np.arange(8), np.arange(8, 20), np.arange(20, 32)]
    values = [rng.normal(i, i+1, len(j)) for i, j in enumerate(idx)]
    return K, idx, values


@pytest.mark.parametrize("r", [0., .6, 1.])
def test_same_condition_and_variance_invariance(r):
    K, idx, values = panel()
    donor = CoregionalisedGP(rho_grid=(r,)).fit_gram(K, idx, values)
    cand = np.arange(8,40)
    mu, sd = donor.predict_gram(K, cand, return_std=True)
    same = donor.recondition(values)
    np.testing.assert_allclose(same.predict_gram(K,cand,True)[0], mu, rtol=1e-12,atol=1e-12)
    assert same.parameter_snapshot() == donor.parameter_snapshot()
    swapped = [values[0]] + [v[::-1] for v in values[1:]]
    other = donor.recondition(swapped)
    np.testing.assert_array_equal(other.predict_gram(K,cand,True)[1], sd)
    np.testing.assert_array_equal(donor.predict_gram(K,cand,True)[0],mu)
    a = acquisition_scores("ei", mu, sd, best=max(values[0]),rng=np.random.default_rng(9))
    b = acquisition_scores("ei", *same.predict_gram(K,cand,True),best=max(values[0]),rng=np.random.default_rng(9))
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(np.argsort(a),np.argsort(b))


def test_zero_covariance_fixed_single_task_parameters():
    K, idx, values = panel()
    stl = TanimotoGP().fit_gram(K[np.ix_(idx[0],idx[0])],values[0])
    donor=CoregionalisedGP(rho_grid=(0.,),eta_grid=(stl.eta_,),fixed_scale=stl.scale_).fit_gram(K,idx,values)
    c=np.arange(8,40)
    want=stl.predict_gram(K[np.ix_(c,idx[0])],return_std=True)
    got=donor.predict_gram(K,c,True)
    np.testing.assert_allclose(got,want,rtol=1e-11,atol=1e-11)


def test_history_permutation_preserves_actual_observed_multisets():
    rng=np.random.default_rng(4)
    vals=np.arange(300).reshape(100,3).astype(float)
    idx=[np.array([2,4,7,12]),np.array([30,31,40,43])]
    values,records=shuffle_auxiliary_responses(vals,idx,[1,2],rng,"history")
    for target,loc,v,record in zip([1,2],idx,values,records):
        np.testing.assert_array_equal(np.sort(v),np.sort(vals[loc,target]))
        np.testing.assert_array_equal(v,vals[record["source_indices"],target])
        assert set(record["source_indices"]) == set(loc)


def test_library_scope_exactly_reproduces_legacy_rng():
    vals=np.arange(300).reshape(100,3).astype(float)
    idx=[np.array([2,4,7,12]),np.array([30,31,40,43])]
    g=np.random.default_rng(99)
    expected=[g.permutation(vals[:,j])[i] for i,j in zip(idx,[1,2])]
    got,_=shuffle_auxiliary_responses(vals,idx,[1,2],np.random.default_rng(99),"library")
    for a,b in zip(got,expected):np.testing.assert_array_equal(a,b)


def test_locking_checks_response_shapes_and_does_not_refit(monkeypatch):
    K,idx,values=panel()
    donor=CoregionalisedGP().fit_gram(K,idx,values)
    def forbidden(*a,**k):raise AssertionError("parameter fitting was called")
    monkeypatch.setattr(TanimotoGP,"fit_gram",forbidden)
    donor.recondition(values)
    with pytest.raises(ValueError):donor.recondition(values[:-1])
    with pytest.raises(ValueError):donor.recondition([values[0][:-1]]+values[1:])


def test_state_fitting_accepts_only_observed_primary_values():
    from crossed_auxiliary import state_posteriors
    K,idx,values=panel()
    measured=values[1:]
    permuted=[v[::-1] for v in measured]
    state=state_posteriors(K,idx[0],values[0],idx[1:],measured,permuted,True)
    assert set(state["posteriors"])=={"stl","MM","MP","PM","PP"}
    assert set(state["cand"])==set(range(8,40))
    np.testing.assert_array_equal(state["posteriors"]["MM"][1],state["posteriors"]["MP"][1])
    np.testing.assert_array_equal(state["posteriors"]["PM"][1],state["posteriors"]["PP"][1])


def test_toy_campaign_budget_and_history_eligibility():
    from crossed_auxiliary import run_case
    rng=np.random.default_rng(2)
    x=(rng.random((170,32))<.3).astype(float)
    K=tanimoto_gram(x,x)
    values=rng.normal(size=(170,4))
    result=run_case(K,values,0,["a","b","c","d"],0,0,(14,21),verify=True)
    for arm,run in result["sequential"].items():
        assert len(run["selected"])==21
        assert len(set(run["selected"]))==21
        assert run["selected"][:10]==result["initial"]
        assert set(run["recall"])=={"14","21"}
    assert result["audit"]["clean"]
    import json
    json.dumps(result, allow_nan=False)
