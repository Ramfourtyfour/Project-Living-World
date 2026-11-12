from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.native import World
from training.proposals import context,proposals,BASE_SIZE,CANDIDATE_SIZE,MAX_CANDIDATES
from training.social_arena import Arena,replay_trace
from training.proposal_baselines import ProposalHeuristicPolicy


def test_slate_uses_local_observations_only_and_has_bounded_finite_inputs(monkeypatch):
    with World(42,8) as world:
        o=world.observe(world.ids[0]);before=world.snapshot()
        monkeypatch.setattr(World,"debug_state",lambda *_:(_ for _ in ()).throw(AssertionError("oracle used")))
        x=context(o);slate=proposals(o)
        assert len(x)==BASE_SIZE and 8<=len(slate)<=MAX_CANDIDATES
        assert all(len(p.features)==CANDIDATE_SIZE for p in slate)
        assert all(abs(v)<=16 for v in x)
        contaminated={**deepcopy(o),"seed":9999,"hidden_registry":[{"id":999,"inventory":[999]*5,"trust":100}]}
        for p in contaminated["people"]:p["inventory"]=[999]*5;p["health"]=0;p["policy_hidden"]=[1]*64
        assert context(contaminated)==x and proposals(contaminated)==slate
        assert world.snapshot()==before


def test_affordable_but_costly_trade_is_a_choice_and_not_masked_as_unwise():
    with Arena(100,"trade_bad") as arena:
        o=arena.world.observe(arena.actor);offer=next(f for f in o["offers"] if f["to"]==arena.actor)
        assert o["self"]["inventory"][offer["ask_resource"]]>=offer["ask_quantity"]
        assert o["self"]["inventory"][offer["resource"]]>=3
        assert any(p.kind=="accept" and p.action.target==offer["id"] and p.enabled for p in proposals(o))
        assert any(p.kind=="decline" and p.action.target==offer["id"] for p in proposals(o))


def test_membership_choices_are_not_triggered_or_blocked_by_trust_thresholds():
    with Arena(100,"invite_bad") as arena:
        o=arena.world.observe(arena.actor)
        for trust in (-100,100):
            altered=deepcopy(o)
            for r in altered["relationships"]:r["trust"]=trust
            kinds=[p.kind for p in proposals(altered) if p.enabled]
            assert "leave" in kinds and "accept" in kinds and "decline" in kinds
        assert arena.world.observe(arena.actor)["self"]["group"]==arena.initial_group


def test_setup_and_baseline_membership_outcomes_replay_without_silent_state_edits(tmp_path):
    policy=ProposalHeuristicPolicy()
    for task,change in (("invite_good",True),("invite_bad",False)):
        path=tmp_path/(task+".jsonl.gz")
        with Arena(101,task,horizon=16,trace=path) as arena:
            for _ in range(16):
                o=arena.world.observe(arena.actor);arena.step({arena.actor:policy.action(o)})
            report=arena.report()
            assert bool(report["membership_changes"])==change and report["conserved"]
        assert replay_trace(path)["state_hash"]==report["state_hash"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,reason="Optional learning dependency")
def test_candidate_scorer_has_no_slot_embeddings_and_is_permutation_equivariant():
    import torch
    from training.social_model import SocialModel,tensors
    torch.set_num_threads(1);torch.manual_seed(3)
    with World(42,4) as world:
        ids=list(world.ids);x,c,mask,_=tensors(world.observations(),ids)
        model=SocialModel();permutation=torch.randperm(MAX_CANDIDATES)
        a,value,h=model(x[None],c[None]);b,value2,h2=model(x[None],c[:,permutation][None])
        assert torch.allclose(a[:,:,permutation],b,atol=1e-6,rtol=1e-6)
        assert torch.equal(value,value2) and torch.equal(h,h2)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,reason="Optional learning dependency")
def test_social_checkpoint_retains_individual_hidden_vectors_and_rejects_bad_memory(tmp_path):
    import torch
    from training.social_model import SocialModel,LearnedSocialPolicy,save_artifact
    from training.checkpoint import save_checkpoint,load_checkpoint
    torch.set_num_threads(1);torch.manual_seed(5)
    model=tmp_path/"social.pt";save_artifact(SocialModel(),model,{"test":True})
    a,b=LearnedSocialPolicy(model),LearnedSocialPolicy(model)
    with World(42,8) as world:
        for _ in range(4):world.step(a.actions(world.observations()))
        memory=a.memory_snapshot();assert len(memory["hidden"])==8
        assert memory["hidden"][str(world.ids[0])]!=memory["hidden"][str(world.ids[1])]
        save_checkpoint(world,tmp_path/"checkpoint",a)
        with load_checkpoint(tmp_path/"checkpoint",b) as resumed:
            assert b.memory_snapshot()==memory
            assert a.actions(world.observations())==b.actions(dict(reversed(list(resumed.observations().items()))))
        previous=b.memory_snapshot();poisoned=deepcopy(previous);next(iter(poisoned["hidden"].values()))[0]=2.0
        with pytest.raises(ValueError):b.restore_memory(poisoned)
        assert b.memory_snapshot()==previous
