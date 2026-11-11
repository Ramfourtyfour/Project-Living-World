import json
from pathlib import Path
import random
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.native import Action,World
from training.policies import HeuristicPolicy

def near(world,actor,target):
    for _ in range(100):
        own=world.observe(actor)["self"]
        if (own["x"]-target["x"])**2+(own["y"]-target["y"])**2<=300**2:return
        result=world.step({actor:Action("move",x=target["x"],y=target["y"])})[0]
        assert result["ok"],result
    raise AssertionError("Target was not reached")

def gather(world,actor,resource,quantity=1):
    node=next(n for n in world.observe(actor)["nodes"] if n["resource"]==resource)
    near(world,actor,node)
    while quantity:
        amount=min(2,quantity)
        assert world.step({actor:Action("gather",target=node["id"],quantity=amount)})[0]["ok"]
        quantity-=amount

def cache_for(world,group):
    return next(c for c in world.debug_state()["caches"] if c["group"]==group)

def test_group_founding_is_individual_and_names_support_utf8():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4]
        assert world.step({a:Action("found_group",text="New circle")})[0]["reason"]=="already_in_group"
        assert world.step({a:Action("depart")})[0]["ok"]
        assert world.step({a:Action("found_group",text="Étoile 森")})[0]["ok"]
        group=world.observe(a)["self"]["group"]
        assert group>4 and world.observe(b)["self"]["group"]==1
        near(world,b,world.observe(a)["self"])
        assert world.step({a:Action("invite",target=b)})[0]["ok"]
        offer=world.observe(b)["offers"][0]
        assert world.observe(b)["self"]["group"]==1
        assert world.step({b:Action("decline",target=offer["id"])})[0]["ok"]
        assert world.observe(b)["self"]["group"]==1
        assert world.step({a:Action("invite",target=b)})[0]["ok"]
        offer=world.observe(b)["offers"][0]
        assert world.step({b:Action("accept",target=offer["id"])})[0]["ok"]
        assert world.observe(b)["self"]["group"]==group
        assert world.step({b:Action("rename_group",text="Independent camp")})[0]["ok"]
        with World(snapshot=world.snapshot()) as resumed:
            assert resumed.snapshot()==world.snapshot()
            assert resumed.observe(a)["agreements"][-1]["name"]=="Independent camp"

@pytest.mark.parametrize("name",[" "," leading","trailing ","bad\nname","x"*41])
def test_malformed_group_names_rejected_without_foundation(name):
    with World(42,4) as world:
        a=world.ids[0];world.step({a:Action("depart")})
        assert world.step({a:Action("found_group",text=name)})[0]["reason"]=="invalid_group_name"
        assert world.observe(a)["self"]["group"]==0
        assert len(world.debug_state()["agreements"])==4

def test_storage_contents_are_local_private_and_revoked_on_departure():
    with World(42,8) as world:
        actor,outsider=world.ids[0],world.ids[4]
        cache=cache_for(world,1);gather(world,actor,0,2)
        near(world,actor,cache)
        assert world.step({actor:Action("deposit",target=cache["id"],resource=0,quantity=2)})[0]["ok"]
        own_cache=next(c for c in world.observe(actor)["caches"] if c["id"]==cache["id"])
        assert own_cache["inventory"][0]==2
        # A formerly affiliated person loses current access while retaining beliefs.
        near(world,outsider,cache);world.step({outsider:Action("depart")})
        view=next(c for c in world.observe(outsider)["caches"] if c["id"]==cache["id"])
        assert "inventory" not in view
        before=world.debug_state()["caches"]
        assert world.step({outsider:Action("withdraw",target=cache["id"],resource=0)})[0]["reason"]=="access_denied"
        assert world.debug_state()["caches"]==before
        assert world.step({actor:Action("withdraw",target=cache["id"],resource=0)})[0]["ok"]
        assert world.debug_state()["conserved"]

def test_offscreen_group_renames_are_not_global_knowledge():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4]
        # Moving a known member far away does not make a remote rename observable.
        target={"x":6000,"y":4000}
        for _ in range(12):world.step({a:Action("move",x=target["x"],y=target["y"])})
        assert b not in {p["id"] for p in world.observe(a)["people"]}
        assert world.step({a:Action("rename_group",text="Remote name")})[0]["ok"]
        known=next(g for g in world.observe(b)["agreements"] if g["id"]==1)
        assert known["name"]=="Circle 1"
        near(world,a,world.observe(b)["self"])
        known=next(g for g in world.observe(b)["agreements"] if g["id"]==1)
        assert known["name"]=="Remote name"

def test_requests_can_be_declined_and_never_override_the_recipient():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];near(world,b,world.observe(a)["self"])
        assert world.step({a:Action("request",target=b,resource=0)})[0]["ok"]
        request=world.observe(b)["requests"][0]
        before=world.observe(b)["self"]["inventory"]
        assert world.step({b:Action("answer_request",target=request["id"],quantity=0)})[0]["ok"]
        assert world.observe(b)["requests"][0]["status"]==2
        assert world.observe(b)["self"]["inventory"]==before
        assert world.observe(a)["self"]["group"]==world.observe(b)["self"]["group"]

def test_accepted_supply_request_requires_a_real_transfer_and_updates_only_recipient_trust():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4]
        gather(world,b,0,2);near(world,a,world.observe(b)["self"])
        world.step({a:Action("request",target=b,resource=0,quantity=2)})
        request=world.observe(b)["requests"][0]
        world.step({b:Action("answer_request",target=request["id"],quantity=1)})
        assert world.observe(a)["requests"][0]["status"]==1
        world.step({b:Action("offer",target=a,resource=0,quantity=2)})
        offer=world.observe(a)["offers"][0]
        world.step({a:Action("accept",target=offer["id"])})
        assert world.observe(a)["requests"][0]["status"]==3
        relation=next(r for r in world.observe(a)["relationships"] if r["id"]==b)
        inverse=next(r for r in world.observe(b)["relationships"] if r["id"]==a)
        assert relation["trust"]>0 and inverse["trust"]==0
        assert world.observe(a)["self"]["inventory"][0]==2
        assert world.observe(b)["self"]["inventory"][0]==0
        with World(snapshot=world.snapshot()) as resumed:
            assert resumed.snapshot()==world.snapshot() and resumed.debug_state()["conserved"]

def test_one_unit_cannot_fulfill_two_supply_promises():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];gather(world,b,0);near(world,a,world.observe(b)["self"])
        for _ in range(2):
            world.step({a:Action("request",target=b,resource=0)})
            request=next(r for r in world.observe(b)["requests"] if r["status"]==0)
            world.step({b:Action("answer_request",target=request["id"])})
        world.step({b:Action("offer",target=a,resource=0)})
        world.step({a:Action("accept",target=world.observe(a)["offers"][0]["id"])})
        states=[r["status"] for r in world.observe(a)["requests"]]
        assert states.count(3)==1 and states.count(1)==1

def test_unknown_requests_and_inventory_information_remain_private():
    with World(42,12) as world:
        a,b,remote=world.ids[0],world.ids[4],world.ids[1]
        near(world,a,world.observe(b)["self"])
        world.step({a:Action("request",target=b,resource=1)})
        assert world.observe(a)["requests"] and world.observe(b)["requests"]
        assert world.observe(remote)["requests"]==[]
        request=world.observe(a)["requests"][0]
        assert world.step({remote:Action("answer_request",target=request["id"])})[0]["reason"]=="unknown_request"

def test_cache_construction_consumes_and_embodies_real_materials():
    with World(42,1) as world:
        a=world.ids[0]
        assert world.step({a:Action("place_cache")})[0]["reason"]=="insufficient_materials"
        gather(world,a,2,3);gather(world,a,4,2)
        before=world.observe(a)["self"];count=len(world.debug_state()["caches"])
        result=world.step({a:Action("place_cache",x=before["x"]+200,y=before["y"])})[0]
        assert result["ok"],result
        after=world.observe(a)["self"]
        assert after["inventory"][2]==after["inventory"][4]==0
        assert len(world.debug_state()["caches"])==count+1
        assert world.debug_state()["conserved"]
        with World(snapshot=world.snapshot()) as resumed:assert resumed.snapshot()==world.snapshot()

def test_legacy_save_migration_preserves_resources_identities_and_does_not_spawn_caches():
    with World(42,8,rules_version=1) as legacy:
        original=legacy.snapshot();state=legacy.debug_state()
        with World(snapshot=original) as migrated:
            assert migrated.rules_version==1 and migrated.snapshot()==original
            migrated.upgrade_rules();after=migrated.debug_state()
            assert migrated.rules_version==2 and after["caches"]==[]
            assert migrated.ids==legacy.ids and after["ledger"]==state["ledger"]
            assert all(a["inventory"]==b["inventory"] for a,b in zip(after["people"],state["people"]))
            saved=migrated.snapshot();migrated.upgrade_rules();assert migrated.snapshot()==saved
            with World(snapshot=saved) as resumed:assert resumed.snapshot()==saved

def test_promise_expiration_changes_only_personal_reliability_not_membership():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];near(world,a,world.observe(b)["self"])
        world.step({a:Action("request",target=b,resource=0)})
        request=world.observe(b)["requests"][0]
        world.step({b:Action("answer_request",target=request["id"])})
        for _ in range(61):world.step({})
        assert world.observe(a)["requests"][0]["status"]==4
        relation=next(r for r in world.observe(a)["relationships"] if r["id"]==b)
        inverse=next(r for r in world.observe(b)["relationships"] if r["id"]==a)
        assert relation["trust"]==-2 and relation["broken"]==1 and inverse["trust"]==0
        assert world.observe(a)["self"]["group"]==world.observe(b)["self"]["group"]==1

def test_guard_can_finish_offscreen_without_becoming_remote_tracking():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];body=world.observe(b)["self"]
        near(world,a,{"x":body["x"]+1000,"y":body["y"]})
        world.step({a:Action("request",target=b,resource=3,x=body["x"],y=body["y"])})
        request=world.observe(b)["requests"][0]
        assert world.step({b:Action("answer_request",target=request["id"])})[0]["ok"]
        for _ in range(12):world.step({a:Action("move",x=body["x"]+2300,y=body["y"])})
        assert b not in {p["id"] for p in world.observe(a)["people"]}
        current=world.observe(a)["requests"][0]
        assert current["status"]==1 and current["progress"]==0
        assert world.observe(b)["requests"][0]["status"]==3
        assert "completed_tick" not in current
        near(world,a,{"x":body["x"]+700,"y":body["y"]})
        for _ in range(10):world.step({})
        assert world.observe(a)["requests"][0]["status"]==3

def test_offscreen_task_report_corrects_an_unverified_expiry_belief():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];body=world.observe(b)["self"]
        near(world,a,{"x":body["x"]+1000,"y":body["y"]})
        world.step({a:Action("request",target=b,resource=3,x=body["x"],y=body["y"])})
        request=world.observe(b)["requests"][0];world.step({b:Action("answer_request",target=request["id"])})
        for _ in range(12):world.step({a:Action("move",x=body["x"]+2300,y=body["y"])})
        assert world.observe(b)["requests"][0]["status"]==3
        assert world.observe(a)["requests"][0]["status"]==1
        for _ in range(50):world.step({})
        assert world.observe(a)["requests"][0]["status"]==4
        relation=next(r for r in world.observe(a)["relationships"] if r["id"]==b)
        assert relation["broken"]==1 and relation["trust"]==-2
        assert world.step({b:Action("report_request",target=request["id"])})[0]["reason"]=="recipient_unreachable"
        near(world,b,world.observe(a)["self"])
        assert world.step({b:Action("report_request",target=request["id"])})[0]["ok"]
        assert world.observe(a)["requests"][0]["status"]==3
        relation=next(r for r in world.observe(a)["relationships"] if r["id"]==b)
        assert relation["broken"]==0 and relation["trust"]==2
        with World(snapshot=world.snapshot()) as resumed:assert resumed.snapshot()==world.snapshot()

def test_accepting_a_visit_does_not_move_the_body_automatically():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];near(world,a,world.observe(b)["self"])
        body=world.observe(b)["self"];target={"x":body["x"]+700,"y":body["y"]}
        world.step({a:Action("request",target=b,resource=2,**target)})
        request=world.observe(b)["requests"][0];world.step({b:Action("answer_request",target=request["id"])})
        for _ in range(4):world.step({})
        assert world.observe(b)["self"]["x"]==body["x"] and world.observe(b)["self"]["y"]==body["y"]
        assert world.observe(a)["requests"][0]["status"]==1
        near(world,b,target)
        assert world.observe(a)["requests"][0]["status"]==3

def test_hidden_targets_are_redacted_in_other_witnesses_event_diaries():
    with World(42,12) as world:
        a,b,c=world.ids[0],world.ids[4],world.ids[8]
        body=world.observe(a)["self"]
        for _ in range(12):world.step({b:Action("move",x=body["x"]-1700,y=body["y"]),c:Action("move",x=body["x"]+1700,y=body["y"])})
        assert c not in {p["id"] for p in world.observe(b)["people"]}
        assert world.step({a:Action("inspect",target=c)})[0]["ok"]
        observed=next(e for e in world.observe(b)["events"] if e["actor"]==a and e["type"]=="inspect")
        assert observed["target"]==0
        with World(snapshot=world.snapshot()) as resumed:
            assert resumed.observe(b)["events"]==world.observe(b)["events"]

def test_storage_memories_do_not_refresh_remote_private_stock():
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];cache=cache_for(world,1)
        gather(world,a,0,2);near(world,a,cache)
        world.step({a:Action("deposit",target=cache["id"],resource=0,quantity=2)})
        view=world.observe(b)
        visible=next(c for c in view["caches"] if c["id"]==cache["id"])
        memory=next(m for m in view["place_memory"] if m["id"]==cache["id"])
        assert "inventory" not in visible and memory["inventory_at_observation"][0]==0
        near(world,b,cache)
        assert next(c for c in world.observe(b)["caches"] if c["id"]==cache["id"])["inventory"][0]==2

def test_malformed_named_c_batch_never_partially_executes():
    import ctypes as C
    from training.native import CAction,CActionV2
    with World(42,4) as world:
        a,b=world.ids[:2];before=world.snapshot()
        first=CActionV2(CAction(a,1,9,0,0,0,0,1,0,0),b"")
        invalid=CActionV2(CAction(b,1,12,0,0,0,0,1,0,0),b"x"*64)
        batch=(CActionV2*2)(first,invalid)
        assert world.lib.lw_step_v2(world.handle,batch,2)==-1
        assert world.snapshot()==before

def test_invalid_utf8_group_label_is_rejected_by_the_native_engine():
    import ctypes as C
    from training.native import CAction,CActionV2
    with World(42,4) as world:
        actor=world.ids[0];world.step({actor:Action("depart")})
        action=CActionV2(CAction(actor,2,12,0,0,0,0,1,0,0),b"\xff")
        batch=(CActionV2*1)(action)
        assert world.lib.lw_step_v2(world.handle,batch,1)==0
        result=json.loads(world._text("lw_results"))[0]
        assert result["reason"]=="invalid_group_name" and world.observe(actor)["self"]["group"]==0

@pytest.mark.parametrize("seed",[13,47])
def test_mixed_social_commands_preserve_materials_references_and_resume(seed):
    rng=random.Random(seed);baseline=HeuristicPolicy()
    choices=("depart","found_group","rename_group","invite","offer","trade","accept","decline",
             "request","answer_request","report_request","deposit","withdraw","place_cache")
    with World(seed,16) as world:
        for tick in range(150):
            observations=world.observations();actions={}
            for actor,o in observations.items():
                if not o["self"]["alive"]:continue
                if rng.random()<0.55:
                    actions[actor]=baseline.actions({actor:o})[actor];continue
                kind=rng.choice(choices);target=999999
                if kind in ("invite","offer","trade","request") and o["people"]:target=rng.choice(o["people"])["id"]
                elif kind in ("accept","decline") and o["offers"]:target=rng.choice(o["offers"])["id"]
                elif kind in ("answer_request","report_request") and o["requests"]:target=rng.choice(o["requests"])["id"]
                elif kind in ("deposit","withdraw") and o["caches"]:target=rng.choice(o["caches"])["id"]
                name=f"Circle {seed}-{actor}-{tick}" if rng.random()<0.9 else "bad\nlabel"
                actions[actor]=Action(kind,target=target,x=o["self"]["x"]+200,y=o["self"]["y"],
                                      resource=rng.randrange(-1,6),quantity=rng.choice([-1,0,1,2,4,24,25]),
                                      ask_resource=rng.randrange(-1,6),ask_quantity=rng.randrange(0,5),text=name)
            if tick%25==0:
                with World(snapshot=world.snapshot()) as resumed:
                    assert resumed.snapshot()==world.snapshot()
                    assert resumed.step(actions)==world.step(actions)
                    assert resumed.snapshot()==world.snapshot()
            else:world.step(actions)
            assert world.debug_state()["conserved"]
