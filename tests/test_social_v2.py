"""Versioned scenario, own-cost, dependency-free inference and memory boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.proposals import proposals as old_proposals
from training.proposals_v2 import proposals
from training.social_arena_v2 import Arena,replay_trace
from training.proposal_baselines import ProposalHeuristicPolicy

ROOT=Path(__file__).resolve().parents[1]
TORCH=importlib.util.find_spec("torch") is not None


def test_v2_own_cost_is_factual_and_does_not_change_action_availability():
    with Arena(100,"trade_bad") as arena:
        o=arena.world.observe(arena.actor);old=old_proposals(o);new=proposals(o)
        assert [(p.kind,p.action,p.enabled) for p in old]==[(p.kind,p.action,p.enabled) for p in new]
        offer=next(f for f in o["offers"] if f["to"]==arena.actor)
        accept=next(p for p in new if p.kind=="accept" and p.action.target==offer["id"])
        decline=next(p for p in new if p.kind=="decline" and p.action.target==offer["id"])
        assert accept.enabled and len(accept.features)==64
        r=offer["ask_resource"];inventory=o["self"]["inventory"]
        assert accept.features[56+r]==(inventory[r]-offer["ask_quantity"])/8
        assert decline.features[56+r]==inventory[r]/8
        assert accept.features[52]==o["self"]["hunger" if r==0 else "thirst"]/100
        contaminated=deepcopy(o)
        for p in contaminated["people"]:p["inventory"]=[999]*5;p["hunger"]=99;p["thirst"]=99
        contaminated["hidden_registry"]=[999]*100
        assert proposals(contaminated)==new


def test_setup_replans_after_unseen_collision_and_replays(tmp_path):
    path=tmp_path/"navigation.jsonl.gz"
    with Arena(1139,"invite_good",horizon=1,trace=path) as arena:
        assert arena.world.observe(arena.actor)["self"]["alive"]
        arena.step({arena.actor:ProposalHeuristicPolicy().action(arena.world.observe(arena.actor))})
        report=arena.report()
    assert report["conserved"]
    assert replay_trace(path)["state_hash"]==report["state_hash"]


@pytest.mark.parametrize("seed",[100,101])
def test_reciprocal_help_means_both_people_consume_actual_received_units(seed,tmp_path):
    path=tmp_path/f"help-{seed}.jsonl.gz";teacher=ProposalHeuristicPolicy()
    with Arena(seed,"help_reciprocal",horizon=128,trace=path) as arena:
        a,p=arena.actor,arena.partner
        for _ in range(128):arena.step({a:teacher.action(arena.world.observe(a))})
        report=arena.report()
    for actor,other in ((a,p),(p,a)):
        assert any(u["actor"]==actor and u["source"]==other and u["via"]=="gift" and u["useful_need_reduction"]>=20 for u in report["useful_consumption"])
    assert report["conserved"] and report["scenario_version"]==2
    assert replay_trace(path)["state_hash"]==report["state_hash"]


@pytest.mark.skipif(not TORCH,reason="Optional learning dependency")
def test_v2_export_native_choices_and_checkpoint_continue_with_only_standard_library(tmp_path):
    import torch
    from training.social_model_v2 import SocialModel,LearnedSocialPolicy,save_artifact
    from inference.social_export_v2 import export_policy
    from inference.social_runtime_v2 import NativeSocialPolicy
    from training.checkpoint import save_checkpoint
    from training.native import World
    torch.set_num_threads(1);torch.manual_seed(71)
    source=tmp_path/"source.pt";binary=tmp_path/"native.lwslate"
    save_artifact(SocialModel(),source,{"test":True});export_policy(source,binary)
    a=LearnedSocialPolicy(source)
    with World(42,8) as world,NativeSocialPolicy(binary) as b:
        for _ in range(12):
            observations=world.observations();expected=a.actions(observations)
            assert b.actions(dict(reversed(list(observations.items()))))==expected
            world.step(expected)
        memory=b.memory_snapshot();bad=deepcopy(memory);next(iter(bad["hidden"].values()))[0]=float("nan")
        with pytest.raises(ValueError):b.restore_memory(bad)
        assert b.memory_snapshot()==memory
        save_checkpoint(world,tmp_path/"checkpoint",b)
        expected=b.actions(world.observations())
        world.step(expected);wanted_hash=world.state_hash()
    script='''import json,sys
from inference.social_runtime_v2 import NativeSocialPolicy
from training.checkpoint import load_checkpoint
with NativeSocialPolicy(sys.argv[1]) as policy, load_checkpoint(sys.argv[2],policy) as world:
    world.step(policy.actions(world.observations()))
    assert world.state_hash()==sys.argv[3]
assert "torch" not in sys.modules and "numpy" not in sys.modules
print("native checkpoint continued")
'''
    result=subprocess.run([sys.executable,"-S","-c",script,str(binary),str(tmp_path/"checkpoint"),wanted_hash],cwd=ROOT,capture_output=True,text=True,check=True)
    assert "native checkpoint continued" in result.stdout


def test_version_dispatch_rejects_mismatched_architecture_without_loading_runtime(tmp_path):
    from tools.social_loading import proposal_version
    source=tmp_path/"source.pt";manifest=source.with_suffix(".json")
    manifest.write_text(json.dumps({"proposal_version":2,"architecture":"social-gru64-slate-v2"}))
    assert proposal_version(source)==2
    manifest.write_text(json.dumps({"proposal_version":2,"architecture":"social-gru64-slate-v1"}))
    with pytest.raises(ValueError):proposal_version(source)
