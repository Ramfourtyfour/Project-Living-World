import ctypes as C
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.native import Action, CAction, World
from training.policies import HeuristicPolicy
from training.env import ParallelWorldEnv
from tools.rollout import record,replay

def approach(w,actor,target):
    for _ in range(200):
        o=w.observe(actor);a=o["self"]
        if (a["x"]-target["x"])**2+(a["y"]-target["y"])**2<=300**2:
            return
        result=w.step({actor:Action("move",x=target["x"],y=target["y"])})
        assert result[0]["ok"]
    raise AssertionError("Could not reach target")

def gather(w,actor,resource,quantity=1):
    n=next(n for n in w.observe(actor)["nodes"] if n["resource"]==resource)
    approach(w,actor,n)
    assert w.step({actor:Action("gather",target=n["id"],quantity=quantity)})[0]["ok"]

@pytest.mark.parametrize("seed",[0,1,42,123456789])
def test_deterministic_save_resume_and_policy(seed):
    policy=HeuristicPolicy()
    with World(seed,32) as a,World(seed,32) as b:
        assert a.state_hash()==b.state_hash()
        for _ in range(32):
            actions=policy.actions(a.observations());a.step(actions);b.step(dict(reversed(list(actions.items()))))
        assert a.state_hash()==b.state_hash()
        saved=a.snapshot()
        with World(snapshot=saved) as c:
            assert c.snapshot()==saved
            for _ in range(32):
                actions=policy.actions(a.observations());a.step(actions);c.step(actions)
                assert a.state_hash()==c.state_hash()

def test_reads_are_pure_and_hidden_world_stays_hidden():
    with World(42,32) as w:
        state=w.debug_state();actor=w.ids[0];o=w.observe(actor);before=w.state_hash()
        assert len(o["people"])<31 and len(o["nodes"])<len(state["nodes"])
        for p in o["people"]:
            assert not set(p)&{"health","inventory","hunger","thirst","fatigue","last_request","style"}
        visible={n["id"] for n in o["nodes"]}
        hidden=next(n for n in state["nodes"] if n["id"] not in visible)
        w.observe(actor);assert before==w.state_hash()
        result=w.step({actor:Action("gather",target=hidden["id"])})
        assert result[0]["reason"]=="not_visible"
        other=next(i for i in w.ids if i not in {p["id"] for p in o["people"]} and i!=actor)
        assert not any(e["actor"]==actor and e["type"]=="rejected" for e in w.observe(other)["events"])

def test_resource_conservation_with_random_and_malformed_actions():
    rng=random.Random(17)
    with World(17,32) as w:
        for _ in range(150):
            observations=w.observations();actions={}
            for i,o in observations.items():
                if not o["self"]["alive"]:continue
                if o["nodes"]:
                    node=rng.choice(o["nodes"])
                    actions[i]=Action(rng.choice(["gather","consume","craft","move"]),target=node["id"],
                                      x=node["x"],y=node["y"],quantity=rng.choice([-10,0,1,2,100000]),resource=rng.randrange(-2,8))
            w.step(actions)
            assert w.debug_state()["conserved"]
        ids=w.ids
        assert len(set(ids))==32

def test_duplicate_requests_and_duplicate_actor_batch():
    with World(1,4) as w:
        actor=w.ids[0]
        assert w.step({actor:Action("depart")},requests={actor:1})[0]["ok"]
        assert w.step({actor:Action()},requests={actor:1})[0]["reason"]=="duplicate_request"
        batch=(CAction*2)(CAction(actor,2,0,0,0,0,0,1,0,0),CAction(actor,3,0,0,0,0,0,1,0,0))
        assert w.lib.lw_step(w.handle,batch,2)==0
        results=json.loads(w._text("lw_results"))
        assert all(r["reason"]=="duplicate_actor_in_batch" for r in results)
        assert w.observe(actor)["self"]["last_request"]==1

def test_depart_and_join_require_individual_decision():
    with World(1,8) as w:
        a,b=w.ids[0],w.ids[4]
        assert w.step({a:Action("depart")})[0]["ok"]
        approach(w,a,w.observe(b)["self"])
        assert w.step({b:Action("invite",target=a)})[0]["ok"]
        offer=w.observe(a)["offers"][0]
        assert w.observe(a)["self"]["group"]==0
        with World(snapshot=w.snapshot()) as resumed:
            assert resumed.step({a:Action("accept",target=offer["id"])})[0]["ok"]
            assert resumed.observe(a)["self"]["group"]==resumed.observe(b)["self"]["group"]
            assert resumed.step({a:Action("accept",target=offer["id"])})[0]["reason"]=="unknown_offer"

def test_atomic_trade_and_repeated_accept():
    with World(2,8) as w:
        a,b=w.ids[0],w.ids[4]
        gather(w,a,0,2);gather(w,b,1,2)
        approach(w,b,w.observe(a)["self"])
        assert w.step({a:Action("trade",target=b,resource=0,quantity=2,ask_resource=1,ask_quantity=2)})[0]["ok"]
        offer=w.observe(b)["offers"][0]
        assert w.step({b:Action("accept",target=offer["id"])})[0]["ok"]
        assert w.observe(a)["self"]["inventory"][:2]==[0,2]
        assert w.observe(b)["self"]["inventory"][:2]==[2,0]
        before=[w.observe(i)["self"]["inventory"] for i in (a,b)]
        assert not w.step({b:Action("accept",target=offer["id"])})[0]["ok"]
        assert before==[w.observe(i)["self"]["inventory"] for i in (a,b)]
        assert w.debug_state()["conserved"]

def test_spent_offer_cannot_mint_items():
    with World(2,8) as w:
        a,b=w.ids[0],w.ids[4];gather(w,a,0);approach(w,b,w.observe(a)["self"])
        w.step({a:Action("offer",target=b,resource=0)})
        offer=w.observe(b)["offers"][0]
        w.step({a:Action("consume",resource=0)})
        assert w.step({b:Action("accept",target=offer["id"])})[0]["reason"]=="insufficient_items"
        assert w.observe(b)["self"]["inventory"][0]==0
        assert w.debug_state()["conserved"]

def test_corrupt_or_wrong_version_saves_rejected():
    with World(1,4) as w:
        saved=w.snapshot()
    for text in ["",saved[:-1],saved.replace("LIVING_WORLD_V1","LIVING_WORLD_V99"),saved+"garbage"]:
        with pytest.raises(ValueError):World(snapshot=text)

def rechecksum(body):
    h=14695981039346656037
    for byte in body.encode("utf-8"):
        h=((h^byte)*1099511628211)&((1<<64)-1)
    return body+f"CHECKSUM {h:016x}\n"

def test_structural_validation_even_with_valid_checksum():
    with World(1,4) as w:
        body=w.snapshot().split("CHECKSUM ")[0]
    lines=body.splitlines(keepends=True)
    # Root number of individuals is bounded even for an attacker who recomputes the checksum.
    assert lines[5].strip()=="4"
    poisoned=lines.copy();poisoned[5]="1000000\n"
    with pytest.raises(ValueError):World(snapshot=rechecksum("".join(poisoned)))
    # Duplicating an ID inside the first individual's remembered sightings is rejected.
    poisoned=body.replace("LIVING_WORLD_V1","LIVING_WORLD_V999",1)
    with pytest.raises(ValueError):World(snapshot=rechecksum(poisoned))
    # Same total material count cannot hide a negative inventory.
    poisoned=lines.copy();poisoned[7]="-1 1 0 0 0\n"
    with pytest.raises(ValueError):World(snapshot=rechecksum("".join(poisoned)))

def test_exhausted_nodes_reject_without_free_materials():
    with World(7,1) as w:
        actor=w.ids[0];stone=next(n for n in w.observe(actor)["nodes"] if n["resource"]==3)
        approach(w,actor,stone)
        # A full inventory is enforced independently of the remaining node capacity.
        for _ in range(12):assert w.step({actor:Action("gather",target=stone["id"],quantity=2)})[0]["ok"]
        assert w.step({actor:Action("gather",target=stone["id"])})[0]["reason"]=="inventory_full"
        assert w.observe(actor)["self"]["inventory"][3]==24
        assert w.debug_state()["conserved"]

def test_expired_offer_cannot_be_accepted():
    with World(2,8) as w:
        a,b=w.ids[0],w.ids[4];gather(w,a,0);approach(w,b,w.observe(a)["self"])
        w.step({a:Action("offer",target=b,resource=0)})
        offer=w.observe(b)["offers"][0]
        for _ in range(41):w.step({})
        assert w.step({b:Action("accept",target=offer["id"])})[0]["reason"]=="unknown_offer"
        assert w.observe(a)["self"]["inventory"][0]==1
        assert w.observe(b)["self"]["inventory"][0]==0

def test_parallel_api_and_rewards():
    with ParallelWorldEnv(population=4,max_cycles=3) as env:
        observations,infos=env.reset(seed=42)
        assert set(observations)==set(infos)==set(env.agents)
        for _ in range(3):
            old=set(env.agents)
            observations,rewards,done,truncated,infos=env.step({i:4 for i in env.agents})
            assert set(rewards)==set(done)==set(truncated)==set(infos)==old
        assert not env.agents
        assert all(truncated.values())

def test_scarcity_is_generated_once_and_persists():
    with World(42,32,scenario=0) as normal,World(42,32,scenario=1) as scarce:
        assert normal.state_hash()!=scarce.state_hash()
        assert scarce.debug_state()["ledger"]["initial"][0]<normal.debug_state()["ledger"]["initial"][0]
        with World(snapshot=scarce.snapshot()) as resumed:
            assert resumed.snapshot()==scarce.snapshot()
            for _ in range(32):
                actions=HeuristicPolicy().actions(scarce.observations());scarce.step(actions);resumed.step(actions)
            assert resumed.state_hash()==scarce.state_hash()
            assert resumed.debug_state()["conserved"]
    with pytest.raises(ValueError):World(42,32,scenario=2)

def test_full_record_replay(tmp_path):
    path=tmp_path/"rollout.jsonl"
    summary=record(path,42,16,32)
    result=replay(path)
    assert result["verified_ticks"]==32 and result["hash"]==summary["hash"]

@pytest.mark.parametrize("seed",[101,202,303,404])
def test_baseline_can_survive_from_empty_inventory(seed):
    policy=HeuristicPolicy()
    with World(seed,32) as w:
        assert all(sum(w.observe(i)["self"]["inventory"])==0 for i in w.ids)
        for _ in range(256):w.step(policy.actions(w.observations()))
        state=w.debug_state()
        assert sum(p["alive"] for p in state["people"])>=28
        assert state["conserved"]
