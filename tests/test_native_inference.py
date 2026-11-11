import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from inference.runtime import NativeLearnedPolicy

def simple_artifact(path):
    inputs,hidden=22,48;count=hidden*inputs+6*hidden*hidden+16*hidden+9
    weights=[0.0]*count;weights[-hidden-1-8+1]=1.0
    data=struct.pack("<8s5I",b"LWGRU01\0",1,inputs,hidden,8,count)+struct.pack(f"<{count}f",*weights)
    path.write_bytes(data)
    manifest={"version":1,"format":"lwgru-v1","architecture":"gru48-v1","feature_version":1,
              "inputs":inputs,"hidden":hidden,"actions":8,"sha256":hashlib.sha256(data).hexdigest()}
    path.with_name(path.name+".json").write_text(json.dumps(manifest))

def test_dependency_free_masks_memory_and_invalid_batches(tmp_path):
    path=tmp_path/"model.lwgru";simple_artifact(path)
    with NativeLearnedPolicy(path) as policy:
        choices,_,_=policy.decide_features([41,42],[[0.0]*22]*2,[[1]*8]*2)
        assert choices==[1,1]
        memory=policy.memory_snapshot()
        with pytest.raises(ValueError,match="duplicate"):
            policy.decide_features([41,41],[[0.0]*22]*2,[[1]*8]*2)
        assert policy.memory_snapshot()==memory
        with pytest.raises(ValueError,match="feature"):
            policy.decide_features([41],[[float("nan")]*22],[[1]*8])
        assert policy.memory_snapshot()==memory
        choices,_,_=policy.decide_features([41],[[0.0]*22],[[1,0,0,0,0,0,0,0]])
        assert choices==[0]
        with pytest.raises(ValueError,match="empty"):
            policy.decide_features([41],[[0.0]*22],[[0]*8])

def test_corrupt_or_truncated_native_artifacts_fail_before_inference(tmp_path):
    path=tmp_path/"model.lwgru";simple_artifact(path)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError,match="checksum"):NativeLearnedPolicy(path)
    manifest_path=path.with_name(path.name+".json");manifest=json.loads(manifest_path.read_text())
    manifest["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest();manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="truncated"):NativeLearnedPolicy(path)

def test_utf8_model_paths_and_recurrent_restore_validation(tmp_path):
    path=tmp_path/"mémoire世界.lwgru";simple_artifact(path)
    with NativeLearnedPolicy(path) as policy:
        policy.decide_features([41],[[0.0]*22],[[1]*8]);before=policy.memory_snapshot()
        broken={**before,"hidden":{"41":[float("nan")]*48}}
        with pytest.raises(ValueError,match="recurrent"):policy.restore_memory(broken)
        assert policy.memory_snapshot()==before
        changed={**before,"runtime_sha256":"0"*64}
        with pytest.raises(ValueError,match="runtime differs"):policy.restore_memory(changed)

@pytest.mark.skipif(importlib.util.find_spec("torch") is None,reason="Optional PyTorch numerical reference")
@pytest.mark.parametrize("architecture",["gru48-v1","gru64-local-v2"])
def test_recurrent_native_outputs_match_pytorch_under_reordered_batches(tmp_path,architecture):
    import torch
    from inference.export import export_policy
    from training.learn import RecurrentPolicy,save_artifact
    torch.set_num_threads(1);torch.manual_seed(91)
    model=RecurrentPolicy(architecture).eval();source=tmp_path/"model.pt";target=tmp_path/"model.lwgru"
    save_artifact(model,source,{"test":True});export_policy(source,target)
    rng=random.Random(7);previous={}
    with NativeLearnedPolicy(target) as native:
        for tick in range(64):
            ids=[41,42,43,44,45,46] if tick%3 else [46,44,42]
            if tick%2:ids.reverse()
            rows=[[rng.uniform(-1,1) for _ in range(native.inputs)] for _ in ids]
            masks=[[1,*[rng.randrange(2) for _ in range(7)]] for _ in ids]
            hidden=torch.stack([previous.get(i,torch.zeros(native.hidden_size)) for i in ids]).unsqueeze(0)
            with torch.no_grad():logits,values,state=model(torch.tensor(rows).unsqueeze(0),hidden)
            choices,actual,actual_values=native.decide_features(ids,rows,masks)
            assert torch.allclose(torch.tensor(actual),logits[0],atol=2e-5,rtol=1e-5)
            assert torch.allclose(torch.tensor(actual_values),values[0],atol=2e-5,rtol=1e-5)
            expected=logits[0].masked_fill(~torch.tensor(masks,dtype=torch.bool),-1e9).argmax(-1).tolist()
            assert choices==expected
            previous.update({i:state[0,j].clone() for j,i in enumerate(ids)})
            snapshot=native.memory_snapshot()
            for i,h in previous.items():assert torch.allclose(torch.tensor(snapshot["hidden"][str(i)]),h,atol=2e-5,rtol=1e-5)

def test_native_world_and_checkpoint_run_without_site_packages(tmp_path):
    artifact=tmp_path/"model.lwgru";simple_artifact(artifact)
    program="""import sys
from inference.runtime import NativeLearnedPolicy
from training.native import World
from training.checkpoint import save_checkpoint,load_checkpoint
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
with NativeLearnedPolicy(sys.argv[1]) as p,World(42,8) as w:
    for _ in range(16):w.step(p.actions(w.observations()))
    save_checkpoint(w,sys.argv[2],p)
    with NativeLearnedPolicy(sys.argv[1]) as q,load_checkpoint(sys.argv[2],q) as restored:
        assert w.state_hash()==restored.state_hash()
        assert p.memory_snapshot()==q.memory_snapshot()
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print('Dependency-free native inference/checkpoint verified')
"""
    result=subprocess.run([sys.executable,"-S","-c",program,str(artifact),str(tmp_path/"checkpoint")],
                          cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "Dependency-free" in result.stdout
