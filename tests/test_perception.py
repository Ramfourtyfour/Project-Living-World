from pathlib import Path
import sys
import random

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.native import World,Action
from training.policies import HeuristicPolicy
from training.policies import move_toward
from training.social import SocialHeuristicPolicy
from training.journal import JournalRunner


def turn(world,actor,x,y):
    assert world.step({actor:Action("turn",x=x,y=y)})[0]["ok"]


def away(world,actor,target):
    a=world.observe(actor)["self"]
    turn(world,actor,max(0,min(24000,a["x"]-(target["x"]-a["x"])*5)),max(0,min(24000,a["y"]-(target["y"]-a["y"])*5)))


def test_turning_reveals_new_sight_without_remotely_refreshing_old_resources():
    with World(42,8,rules_version=3) as world:
        a=world.ids[0]
        turn(world,a,2000,4000)
        food=next(n for n in world.observe(a)["nodes"] if n["resource"]==0)
        assert not any(n["resource"]==1 for n in world.observe(a)["nodes"])
        old=next(m for m in world.observe(a)["memory"] if m["id"]==food["id"])
        turn(world,a,6000,4000)
        observation=world.observe(a)
        assert any(n["resource"]==1 for n in observation["nodes"])
        assert food["id"] not in {n["id"] for n in observation["nodes"]}
        memory=next(m for m in observation["memory"] if m["id"]==food["id"])
        assert memory["seen_tick"]==old["seen_tick"] and memory["age"]>old["age"]
        before=world.snapshot()
        for _ in range(3):world.observe(a)
        assert world.snapshot()==before


def test_hearing_has_no_identity_exact_source_position_or_private_details():
    with World(42,8,rules_version=3) as world:
        a,b=world.ids[0],world.ids[4];source=world.observe(b)["self"]
        away(world,a,source)
        assert b not in {p["id"] for p in world.observe(a)["people"]}
        old=world.observe(a)["person_memory"]
        assert world.step({b:Action("call")})[0]["ok"]
        o=world.observe(a);cue=o["heard"][-1]
        assert cue["kind"]=="call" and cue["source"]=="hearing"
        assert set(cue)=={"local_id","observed_tick","age","source","kind","listener_origin","bearing_octant","bearing_half_width_degrees","distance_interval_cm","confidence"}
        assert 0<=cue["bearing_octant"]<8 and cue["bearing_half_width_degrees"]==23
        assert o["person_memory"]==[{**m,"age":m["age"]+1,"confidence":max(10,m["confidence"]-1)} for m in old]
        assert b not in {p["id"] for p in o["people"]}
        for _ in range(41):world.step({})
        assert world.observe(a)["heard"]==[]


def test_unseen_known_person_cannot_be_inspected_or_requested_through_fov():
    with World(42,8,rules_version=3) as world:
        a,b=world.ids[0],world.ids[4];body=world.observe(b)["self"]
        away(world,a,body)
        assert world.step({a:Action("inspect",target=b)})[0]["reason"]=="not_visible"
        assert world.step({a:Action("request",target=b,resource=0)})[0]["reason"]=="not_visible"
        assert world.observe(b)["requests"]==[]
        turn(world,a,body["x"],body["y"])
        assert world.step({a:Action("request",target=b,resource=0)})[0]["ok"]
        own=world.observe(a)["self"];turn(world,b,own["x"],own["y"])
        request=world.observe(b)["requests"][0]
        assert world.step({b:Action("answer_request",target=request["id"],quantity=0)})[0]["ok"]
        assert world.observe(a)["requests"][0]["status"]==2


def test_person_sightings_remain_dated_beliefs_when_someone_moves_away():
    with World(42,8,rules_version=3) as world:
        a,b=world.ids[0],world.ids[4];body=world.observe(b)["self"]
        turn(world,a,body["x"],body["y"])
        sighting=next(m for m in world.observe(a)["person_memory"] if m["id"]==b)
        away(world,a,body)
        for _ in range(8):world.step({b:Action("move",x=6000,y=4000)})
        memory=next(m for m in world.observe(a)["person_memory"] if m["id"]==b)
        assert (memory["x"],memory["y"],memory["observed_tick"])==(sighting["x"],sighting["y"],sighting["observed_tick"])
        assert memory["age"]>0 and memory["confidence"]<100
        assert "inventory" not in memory and "health" not in memory
        with World(snapshot=world.snapshot()) as resumed:assert resumed.observe(a)==world.observe(a)


@pytest.mark.parametrize("old_rules",[1,2])
def test_explicit_perception_upgrade_preserves_original_people_materials_and_history(old_rules):
    with World(42,8,rules_version=old_rules) as old:
        old.step({old.ids[0]:Action("depart")});original=old.snapshot();state=old.debug_state()
        with World(snapshot=original) as new:
            new.upgrade_rules(3)
            assert new.rules_version==3 and new.ids==old.ids
            assert new.debug_state()["ledger"]==state["ledger"] and new.debug_state()["cover"]==[]
            assert new.observe(new.ids[0])["self"]["group"]==0
            assert len(new.debug_state()["caches"])==len(state.get("caches",[]))
            after=new.snapshot();new.upgrade_rules(3);assert new.snapshot()==after
            with pytest.raises(ValueError):new.upgrade_rules(2)
        assert old.snapshot()==original


def test_perception_and_sound_survive_a_durable_mixed_action_replay(tmp_path):
    path=tmp_path/"perception.lwj";rng=random.Random(7);policy=HeuristicPolicy()
    with JournalRunner.create(path,42,16,policy=policy,rules_version=3) as runner:
        for tick in range(100):
            actions=policy.actions(runner.world.observations())
            actor=rng.choice(runner.world.ids)
            if tick%3==0:actions[actor]=Action("call")
            elif tick%3==1:actions[actor]=Action("crouch",quantity=tick%2)
            runner.advance(actions)
            assert runner.world.debug_state()["conserved"]
        expected=runner.world.snapshot();observations=runner.world.observations()
    with JournalRunner.open(path,HeuristicPolicy()) as verified:
        assert verified.world.snapshot()==expected and verified.world.observations()==observations


def test_deceased_people_receive_no_current_senses_or_new_hearing():
    with World(42,8,rules_version=3) as world:
        for _ in range(160):world.step({})
        a,b=world.ids[0],world.ids[4];o=world.observe(a)
        assert not o["self"]["alive"]
        assert o["senses"]["ambient_light"] is None and o["senses"]["cover_density"] is None
        assert not o["nodes"] and not o["people"] and not o["heard"]
        assert world.step({b:Action("call")})[0]["reason"]=="deceased"
        assert not world.observe(a)["heard"]


def test_fresh_world_social_baseline_can_turn_reply_and_transfer_real_food():
    with World(42,8) as world:
        assert world.rules_version==3
        a,b=world.ids[0],world.ids[4];turn(world,b,2000,4000)
        node=next(n for n in world.observe(b)["nodes"] if n["resource"]==0)
        for _ in range(20):
            o=world.observe(b);own=o["self"]
            if (own["x"]-node["x"])**2+(own["y"]-node["y"])**2<=300**2:break
            world.step({b:move_toward(o,node["x"],node["y"])})
        for _ in range(2):assert world.step({b:Action("gather",target=node["id"],quantity=2)})[0]["ok"]
        body=world.observe(b)["self"]
        for _ in range(20):
            o=world.observe(a);own=o["self"]
            if (own["x"]-body["x"])**2+(own["y"]-body["y"])**2<=300**2:break
            world.step({a:move_toward(o,body["x"],body["y"])})
        turn(world,a,body["x"],body["y"])
        assert world.step({a:Action("request",target=b,resource=0,quantity=2)})[0]["ok"]
        policy=SocialHeuristicPolicy()
        for _ in range(24):
            world.step({i:policy.action(world.observe(i)) for i in (a,b)})
            if world.observe(a)["requests"][0]["status"]==3:break
        assert world.observe(a)["requests"][0]["status"]==3
        assert world.observe(a)["self"]["inventory"][0]>=2 and world.debug_state()["conserved"]
