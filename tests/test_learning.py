"""Learning-interface tests do not pretend a trained policy is intelligent."""
import importlib.util
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

pytestmark=pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None,
                            reason="Optional training dependencies are not installed")

def test_individual_recurrent_memory_and_manifest_roundtrip(tmp_path):
    import torch
    from training.learn import LearnedPolicy,RecurrentPolicy,save_artifact
    from training.native import World
    torch.set_num_threads(1);torch.manual_seed(1)
    path=tmp_path/"model.pt";save_artifact(RecurrentPolicy(),path,{"test":True})
    a=LearnedPolicy(path);b=LearnedPolicy(path)
    with World(42,4) as w:
        observations=w.observations();a.intents(observations)
        saved=a.memory_snapshot();b.restore_memory(saved)
        assert a.intents(observations)==b.intents(observations)
        assert set(a.hidden)==set(w.ids)
        assert not torch.equal(a.hidden[w.ids[0]],a.hidden[w.ids[1]])
    altered=path.read_bytes()+b"tampered";path.write_bytes(altered)
    with pytest.raises(ValueError,match="checksum"):LearnedPolicy(path)

def test_features_and_masks_use_only_observation():
    from training.native import World
    from training.policies import features,intent_mask
    with World(42,4) as w:
        obs=w.observe(w.ids[0]);f=features(obs);mask=intent_mask(obs)
        assert len(f)==22 and len(mask)==8
        assert all(isinstance(v,(int,float)) for v in f)
        assert not mask[2] and not mask[3]
