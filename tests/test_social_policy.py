from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.native import Action,World
from training.social import SocialHeuristicPolicy
from test_social import gather,near

def test_social_baseline_accepts_help_with_surplus_and_declines_without_it():
    policy=SocialHeuristicPolicy()
    for stock,expected in ((4,1),(1,2)):
        with World(42,8) as world:
            a,b=world.ids[0],world.ids[4];gather(world,b,0,stock);near(world,a,world.observe(b)["self"])
            world.step({a:Action("request",target=b,resource=0,quantity=2)})
            own=world.observe(b)
            action=policy.action(own)
            assert action.kind=="answer_request" and action.quantity==(expected==1)
            world.step({b:action})
            assert world.observe(a)["requests"][0]["status"]==expected

def test_social_baseline_fulfills_by_ordinary_offer_and_acceptance():
    policy=SocialHeuristicPolicy()
    with World(42,8) as world:
        a,b=world.ids[0],world.ids[4];gather(world,b,0,4);near(world,a,world.observe(b)["self"])
        world.step({a:Action("request",target=b,resource=0,quantity=2)})
        for _ in range(24):
            world.step({i:policy.action(world.observe(i)) for i in (a,b)})
            if world.observe(a)["requests"][0]["status"]==3:break
        assert world.observe(a)["requests"][0]["status"]==3
        assert any(e["type"]=="gift" for e in world.observe(a)["events"])
        assert world.debug_state()["conserved"]
