import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.checkpoint import load_checkpoint,save_checkpoint
from training.native import World
from training.policies import HeuristicPolicy

def test_checkpoint_is_immutable_and_never_loads_an_incomplete_bundle(tmp_path):
    path=tmp_path/"complete"
    with World(42,8) as world:
        metadata=save_checkpoint(world,path,HeuristicPolicy())
        with load_checkpoint(path,HeuristicPolicy()) as resumed:
            assert resumed.snapshot()==world.snapshot()
        with pytest.raises(FileExistsError):save_checkpoint(world,path)
        assert json.loads((path/"manifest.json").read_bytes())==metadata
    incomplete=tmp_path/"interrupted";incomplete.mkdir();(incomplete/"world.lws").write_text("partial")
    with pytest.raises(ValueError,match="incomplete"):load_checkpoint(incomplete)

def test_checkpoint_rejects_corruption_and_runtime_mismatch(tmp_path):
    path=tmp_path/"world"
    with World(42,4) as world:save_checkpoint(world,path)
    manifest=path/"manifest.json";metadata=json.loads(manifest.read_bytes())
    original=metadata.copy();metadata["native_sha256"]="0"*64;manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="runtime differs"):load_checkpoint(path)
    manifest.write_text(json.dumps(original))
    with (path/"world.lws").open("ab") as stream:stream.write(b"corrupt")
    with pytest.raises(ValueError,match="integrity mismatch"):load_checkpoint(path)

@pytest.mark.parametrize("metadata",[{},[],{"version":1,"files":None}])
def test_malformed_checkpoint_contract_is_rejected_cleanly(tmp_path,metadata):
    (tmp_path/"manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):load_checkpoint(tmp_path)

@pytest.mark.skipif(importlib.util.find_spec("torch") is None,reason="Optional neural dependency")
def test_neural_world_checkpoint_replays_with_individual_memory(tmp_path):
    import torch
    from training.learn import LearnedPolicy,RecurrentPolicy,save_artifact
    torch.set_num_threads(1);torch.manual_seed(7)
    artifact=tmp_path/"policy.pt";save_artifact(RecurrentPolicy("gru64-local-v2"),artifact,{"test":True})
    first=LearnedPolicy(artifact);second=LearnedPolicy(artifact)
    path=tmp_path/"checkpoint"
    with World(42,8) as world:
        for _ in range(12):world.step(first.actions(world.observations()))
        save_checkpoint(world,path,first)
        with pytest.raises(ValueError,match="requires"):load_checkpoint(path)
        with load_checkpoint(path,second) as resumed:
            assert first.memory_snapshot()==second.memory_snapshot()
            for _ in range(24):
                a=first.actions(world.observations());b=second.actions(resumed.observations())
                assert a==b
                world.step(a);resumed.step(b)
                assert world.state_hash()==resumed.state_hash()
        # Even a recomputed file digest cannot authorize memory for a missing identity.
        record=json.loads((path/"controller.json").read_bytes())
        record["memory"]["hidden"]["999999"]=[0.0]*64
        data=json.dumps(record).encode();(path/"controller.json").write_bytes(data)
        manifest=json.loads((path/"manifest.json").read_bytes());manifest["files"]["controller.json"]=hashlib.sha256(data).hexdigest()
        (path/"manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match="unknown individuals"):load_checkpoint(path,second)
