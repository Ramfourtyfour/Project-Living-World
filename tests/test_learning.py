"""Learning-interface tests do not pretend a trained policy is intelligent."""
import importlib.util
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

pytestmark=pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None,
                            reason="Optional training dependencies are not installed")

@pytest.mark.parametrize("architecture",["gru48-v1","gru64-local-v2"])
def test_individual_recurrent_memory_and_manifest_roundtrip(tmp_path,architecture):
    import torch
    from training.learn import LearnedPolicy,RecurrentPolicy,save_artifact
    from training.native import World
    torch.set_num_threads(1);torch.manual_seed(1)
    path=tmp_path/"model.pt";save_artifact(RecurrentPolicy(architecture),path,{"test":True})
    a=LearnedPolicy(path);b=LearnedPolicy(path)
    with World(42,4) as w:
        observations=w.observations();a.intents(observations)
        saved=a.memory_snapshot();b.restore_memory(saved)
        assert a.intents(observations)==b.intents(observations)
        assert set(a.hidden)==set(w.ids)
        assert not torch.equal(a.hidden[w.ids[0]],a.hidden[w.ids[1]])
        assert all(h.numel()==a.model.hidden_size for h in a.hidden.values())
    altered=path.read_bytes()+b"tampered";path.write_bytes(altered)
    with pytest.raises(ValueError,match="checksum"):LearnedPolicy(path)

def test_features_and_masks_use_only_observation():
    from training.native import World
    from training.policies import features,features_v2,intent_mask
    with World(42,4) as w:
        obs=w.observe(w.ids[0]);f=features(obs);mask=intent_mask(obs)
        assert len(f)==22 and len(mask)==8
        assert all(isinstance(v,(int,float)) for v in f)
        assert not mask[2] and not mask[3]
        assert len(features_v2(obs))==38
        assert features_v2({**obs,"hidden_registry":[{"inventory":[999]*5}]})==features_v2(obs)

def test_waste_reward_is_versioned_and_does_not_change_world_rules():
    from training.env import ParallelWorldEnv
    from training.native import Action
    with ParallelWorldEnv(1,512,reward_version=1) as old,ParallelWorldEnv(1,512,reward_version=2) as new:
        for env in (old,new):
            observations,_=env.reset(seed=42);actor=env.agents[0]
            target=next(n for n in observations[actor]["nodes"] if n["resource"]==0)
            for _ in range(8):
                o=env.world.observe(actor);a=o["self"]
                if (a["x"]-target["x"])**2+(a["y"]-target["y"])**2<=300**2:break
                env.step({actor:Action("move",x=target["x"],y=target["y"])})
            env.step({actor:Action("gather",target=target["id"])})
        actor=old.agents[0];need=old.world.observe(actor)["self"]["hunger"]
        _,r1,_,_,i1=old.step({actor:2});_,r2,_,_,i2=new.step({actor:2})
        assert old.world.state_hash()==new.world.state_hash()
        assert "waste" not in i1[actor]["reward_components"]
        assert i2[actor]["reward_components"]["waste"]==pytest.approx(-0.02*max(0,45-need))
        assert r2[actor]-r1[actor]==pytest.approx(i2[actor]["reward_components"]["waste"])
