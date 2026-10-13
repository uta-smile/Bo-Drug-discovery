"""Agentic BO: the policy hook, its controls, and the ceiling.

The load-bearing test here is :func:`test_fixed_policy_matches_no_policy`. Every
comparison this package is for -- LLM controller against scripted control
against ceiling -- is paired on the seed, and pairing only means anything if
adding the hook does not itself move the run. If the hook consumed RNG or took a
different scoring path, every delta measured downstream would be contaminated by
the machinery rather than by the policy, and the whole study would silently
measure nothing.

The rest protect the same properties the constraint arm needed: a policy cannot
see unobserved values, a broken model call cannot crash a sweep or pass silently,
and an unknown acquisition is rejected rather than quietly falling back to ``ei``
(which would make the arm identical to plain BO while still being reported as
an LLM arm).
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.agentic.diagnostics import (
    ACQUISITIONS,
    Diagnostics,
    PolicyDecision,
    _Incumbent,
    build_diagnostics,
)
from molbo.agentic.policies import (
    FixedPolicy,
    LLMPolicy,
    OraclePolicy,
    RandomPolicy,
    RoundRobinPolicy,
    ScriptedPolicy,
)
from molbo.campaign import ArmConfig, run_campaign
from molbo.constraints import LibraryContext
from molbo.llm import Backend, LLMUnavailable, ResponseCache
from molbo.oracles import LookupOracle
from molbo.surrogate import fingerprint_matrix


def test_acquisition_names_match_the_scorer():
    """The policy layer and the scorer must spell the acquisitions identically.

    They are two literal tuples in two modules -- ``agentic.diagnostics`` cannot
    import ``molbo.acquisition`` without pulling scipy and rdkit into what is
    deliberately a numpy-only path. They drifted once: the policy layer said
    ``thompson`` where the scorer says ``ts``, so ``PolicyDecision.validated()``
    waved the name through and ``acquisition_scores`` raised mid-campaign. Every
    policy that ever chose Thompson sampling -- ``ScriptedPolicy`` rule 3,
    ``OraclePolicy``, and any ``LLMPolicy`` taking the option its own prompt
    offered -- crashed the run instead of scoring it.
    """
    from molbo.acquisition import ACQUISITIONS as SCORER_ACQUISITIONS

    assert tuple(ACQUISITIONS) == tuple(SCORER_ACQUISITIONS)

TABLE = {
    "c1ccc2ncnc(Nc3ccccc3)c2c1": 9.0,
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC": 9.5,
    "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1": 8.5,
    "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1": 8.2,
    "c1ccc2ncnc(Nc3ccc(Br)cc3)c2c1": 8.8,
    "c1ccc2ncnc(Nc3cccc(C)c3)c2c1": 7.9,
    "CCO": 4.0,
    "CCCCCCCC": 4.2,
    "c1ccccc1": 5.0,
    "CC(=O)Oc1ccccc1C(=O)O": 5.5,
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O": 4.8,
    "CCCCO": 4.1,
    "c1ccc(Cl)cc1": 5.2,
    "CCN(CC)CC": 4.3,
    "OCC(O)CO": 4.4,
    "c1ccncc1": 5.1,
}


@pytest.fixture(scope="module")
def shared():
    oracle = LookupOracle(TABLE, name="toy")
    lib = oracle.library
    return oracle, lib, LibraryContext.build(lib), fingerprint_matrix(lib)


def run(shared, arm, seed=0, budget=10):
    oracle, lib, ctx, X = shared
    return run_campaign(oracle, arm, seed, budget=budget, library=lib, ctx=ctx, X=X)


def diag(**over) -> Diagnostics:
    base = dict(
        round_index=3,
        calls_spent=30,
        budget=100,
        n_observed=30,
        library_size=2086,
        mu=np.array([0.2, 0.6, 0.4]),
        sd=np.array([0.3, 0.1, 0.2]),
        best_observed=9.0,
        gp_eta=0.05,
        gp_lml=-10.0,
        acquisition="ei",
        rounds_since_improvement=0,
    )
    base.update(over)
    return build_diagnostics(**base)


# ---- the hook must be inert when it changes nothing ------------------------


def test_fixed_policy_matches_no_policy(shared):
    """A policy returning the arm's own settings must reproduce the run exactly.

    This is what makes the paired design valid for every arm in this package.
    """
    plain = ArmConfig("plain", strategy="bo", acquisition="ei", n_init=4)
    hooked = ArmConfig(
        "hooked", strategy="bo", acquisition="ei", n_init=4, policy_source=FixedPolicy("ei")
    )
    for seed in (0, 1, 2):
        a = run(shared, plain, seed=seed)
        b = run(shared, hooked, seed=seed)
        assert a.smiles == b.smiles, f"seed {seed}: the hook changed which molecules were bought"
        assert a.scored == b.scored


def test_policy_log_records_every_round(shared):
    arm = ArmConfig(
        "logged", strategy="bo", acquisition="ei", n_init=4, policy_source=RoundRobinPolicy()
    )
    res = run(shared, arm, budget=10)
    assert res.policy_log, "a policy arm must leave a trace"
    entry = res.policy_log[0]
    assert entry["decision"]["acquisition"] in ACQUISITIONS
    assert "diagnostics" in entry
    assert res.arm_config["has_policy_source"] is True


def test_arm_without_policy_logs_nothing(shared):
    res = run(shared, ArmConfig("plain", strategy="bo", n_init=4))
    assert res.policy_log == []
    assert res.arm_config["has_policy_source"] is False


# ---- the decision boundary -------------------------------------------------


def test_unknown_acquisition_is_rejected_not_defaulted():
    """Falling back to `ei` would make a broken arm look like plain BO."""
    with pytest.raises(ValueError, match="unknown acquisition"):
        PolicyDecision(acquisition="ei_v2").validated()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_parameters_rejected(bad):
    with pytest.raises(ValueError):
        PolicyDecision(acquisition="ei", xi=bad).validated()


def test_negative_beta_rejected():
    with pytest.raises(ValueError, match="beta"):
        PolicyDecision(acquisition="ucb", beta=-1.0).validated()


# ---- diagnostics -----------------------------------------------------------


def test_diagnostics_carry_no_unobserved_values():
    """A controller may see the surrogate's beliefs, never the answer key."""
    d = diag()
    fields = set(d.to_dict())
    for leak in ("values", "true_values", "optimum", "best_possible", "library_values"):
        assert leak not in fields
    assert d.fraction_explored == pytest.approx(30 / 2086)
    assert d.budget_remaining == 70


def test_diagnostics_handle_an_empty_candidate_pool():
    d = diag(mu=np.array([]), sd=np.array([]))
    assert d.n_candidates == 0
    assert d.summary()  # must still render rather than raise


def test_incumbent_tracks_staleness():
    inc = _Incumbent()
    inc.update(1.0, 0)
    inc.update(1.0, 1)
    inc.update(1.0, 2)
    assert inc.stale_for(5) == 5
    inc.update(2.0, 6)
    assert inc.stale_for(6) == 0


# ---- the controls ----------------------------------------------------------


def test_round_robin_cycles():
    p = RoundRobinPolicy()
    seen = [p(diagnostics=diag(round_index=i)).acquisition for i in range(len(ACQUISITIONS) * 2)]
    assert seen[: len(ACQUISITIONS)] == list(ACQUISITIONS)
    assert seen[len(ACQUISITIONS) :] == list(ACQUISITIONS)


def test_random_policy_only_emits_known_acquisitions():
    rng = np.random.default_rng(0)
    p = RandomPolicy()
    for _ in range(50):
        assert p(diagnostics=diag(), rng=rng).acquisition in ACQUISITIONS


def test_random_policy_is_reproducible_from_the_seed():
    p = RandomPolicy()
    a = [p(diagnostics=diag(), rng=np.random.default_rng(7)).acquisition for _ in range(5)]
    b = [p(diagnostics=diag(), rng=np.random.default_rng(7)).acquisition for _ in range(5)]
    assert a == b


def test_scripted_endgame_exploits():
    p = ScriptedPolicy()
    d = p(diagnostics=diag(calls_spent=96, budget=100, rounds_since_improvement=0))
    assert d.acquisition == "greedy"
    assert d.rationale == "endgame"


def test_scripted_explores_when_stalled():
    p = ScriptedPolicy(stale_rounds=3)
    d = p(diagnostics=diag(rounds_since_improvement=4))
    assert d.acquisition == "ucb"
    assert d.beta > 2.0


def test_scripted_reacts_to_a_collapsed_posterior():
    p = ScriptedPolicy(sd_floor=0.5)
    p(diagnostics=diag(sd=np.array([1.0, 1.0, 1.0])))  # sets the reference
    d = p(diagnostics=diag(sd=np.array([0.01, 0.01, 0.01])))
    assert d.acquisition == "ts"


def test_scripted_reset_clears_the_reference():
    """Without reset, one seed's posterior scale would leak into the next and the
    arm would depend on the order seeds were run in."""
    p = ScriptedPolicy(sd_floor=0.5)
    p(diagnostics=diag(sd=np.array([1.0, 1.0, 1.0])))
    p.reset()
    d = p(diagnostics=diag(sd=np.array([0.01, 0.01, 0.01])))
    assert d.acquisition == "ei", "a stale reference survived reset()"


# ---- the ceiling -----------------------------------------------------------


def test_oracle_policy_picks_the_acquisition_that_reaches_the_best_value():
    # `greedy` follows mu, so it takes index 2. Make index 2 the true best and
    # the hindsight ceiling must choose greedy.
    values = np.array([1.0, 2.0, 99.0, 3.0])
    p = OraclePolicy(values=values)
    d = p(
        diagnostics=diag(),
        mu=np.array([0.1, 0.2, 5.0, 0.3]),
        sd=np.array([0.1, 0.1, 0.1, 0.1]),
        cand=np.array([0, 1, 2, 3]),
        best=0.5,
        rng=np.random.default_rng(0),
    )
    assert d.acquisition in ACQUISITIONS
    assert "hindsight" in d.rationale


def test_oracle_policy_survives_an_empty_pool():
    p = OraclePolicy(values=np.array([1.0, 2.0]))
    d = p(
        diagnostics=diag(),
        mu=np.array([]),
        sd=np.array([]),
        cand=np.array([], dtype=int),
        best=0.0,
        rng=np.random.default_rng(0),
    )
    assert d.acquisition in ACQUISITIONS


# ---- the arm under test ----------------------------------------------------


class _Backend(Backend):
    """A scripted stand-in. `payload=None` simulates an unreachable endpoint."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def key(self) -> str:
        return "fake:v1"

    def complete(self, system, prompt, schema):
        self.calls += 1
        if self.payload is None:
            raise LLMUnavailable("endpoint down")
        return dict(self.payload), {"usage": {}}


def test_llm_policy_uses_a_valid_response():
    b = _Backend({"acquisition": "ucb", "xi": 0.02, "beta": 3.5, "rationale": "explore"})
    d = LLMPolicy(backend=b)(diagnostics=diag())
    assert (d.acquisition, d.beta) == ("ucb", 3.5)


def test_llm_policy_falls_back_when_the_endpoint_is_down():
    """A dead endpoint must not crash a 20-seed sweep -- and must not be silent."""
    p = LLMPolicy(backend=_Backend(None))
    d = p(diagnostics=diag())
    assert d.acquisition == "ei"
    assert "fallback" in d.rationale
    assert p.log[-1]["failed"]


def test_llm_policy_falls_back_on_an_invalid_acquisition():
    p = LLMPolicy(backend=_Backend({"acquisition": "made_up", "xi": 0.0, "beta": 1.0}))
    d = p(diagnostics=diag())
    assert d.acquisition == "ei"
    assert "unusable response" in p.log[-1]["failed"]


def test_llm_policy_caches(tmp_path):
    cache = ResponseCache(tmp_path)
    b = _Backend({"acquisition": "pi", "xi": 0.01, "beta": 2.0, "rationale": "r"})
    p = LLMPolicy(backend=b, cache=cache)
    first = p(diagnostics=diag())
    p2 = LLMPolicy(backend=b, cache=cache)
    second = p2(diagnostics=diag())
    assert b.calls == 1, "identical diagnostics must hit the cache"
    assert first.acquisition == second.acquisition == "pi"
    assert p2.log[-1]["cached"] is True


def test_llm_policy_every_n_reuses_the_last_decision():
    b = _Backend({"acquisition": "ucb", "xi": 0.01, "beta": 2.0, "rationale": "r"})
    p = LLMPolicy(backend=b, every=3)
    for i in range(6):
        p(diagnostics=diag(round_index=i))
    assert b.calls == 2, f"every=3 over 6 rounds should call twice, called {b.calls}"


def test_llm_policy_request_key_tracks_the_backend():
    """Two models must not share cache entries."""
    d = diag()
    k1 = LLMPolicy(backend=_Backend({})).request_key_for(d)
    other = _Backend({})
    other.key = lambda: "fake:v2"  # type: ignore[method-assign]
    k2 = LLMPolicy(backend=other).request_key_for(d)
    assert k1 != k2


def test_llm_policy_drives_a_real_campaign(shared):
    b = _Backend({"acquisition": "ucb", "xi": 0.01, "beta": 2.0, "rationale": "explore"})
    arm = ArmConfig(
        "bo_llm_policy", strategy="bo", acquisition="ei", n_init=4, policy_source=LLMPolicy(backend=b)
    )
    res = run(shared, arm, budget=10)
    assert res.n_calls <= 10
    assert all(e["decision"]["acquisition"] == "ucb" for e in res.policy_log)


# ---- the SARA backend ------------------------------------------------------


def test_sara_from_env_names_the_missing_variables(monkeypatch):
    from molbo.agentic.sara import SaraBackend

    monkeypatch.delenv("SARA_BASE_URL", raising=False)
    monkeypatch.delenv("SARA_MODEL", raising=False)
    with pytest.raises(LLMUnavailable, match="SARA_BASE_URL"):
        SaraBackend.from_env()


def test_sara_key_records_sampling_settings():
    from molbo.agentic.sara import SaraBackend

    a = SaraBackend(base_url="http://x", model="m", temperature=0.0)
    b = SaraBackend(base_url="http://x", model="m", temperature=0.7)
    assert a.key() != b.key(), "a sampling change must invalidate the cache"
    assert a.is_deterministic and not b.is_deterministic


def test_sara_strips_a_json_fence():
    from molbo.agentic.sara import _strip_fence

    assert _strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fence('{"a": 1}') == '{"a": 1}'
