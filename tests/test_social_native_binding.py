"""Exercise both native social bindings without optional learning dependencies."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("version,width,count",[(1,52,37258),(2,64,38026)])
def test_dependency_free_versioned_social_binding_and_atomic_memory(tmp_path,version,width,count):
    module=importlib.import_module("inference.social_runtime"+("_v2" if version==2 else ""))
    path=tmp_path/"démo.lwslate"
    data=struct.pack("<8s6I",f"LWSLAT{version}\0".encode(),version,64,width,64,128,count)+bytes(count*4)
    path.write_bytes(data)
    manifest={"version":version,"format":f"lwslate-v{version}","architecture":f"social-gru64-slate-v{version}","observation_version":3,
              "proposal_version":version,"base_inputs":64,"candidate_inputs":width,"hidden":64,"max_candidates":128,
              "candidate_kinds":list(module.KINDS),"sha256":hashlib.sha256(data).hexdigest(),
              "interface_sources":{name:hashlib.sha256((module.ROOT/name).read_bytes()).hexdigest() for name in module.INTERFACE}}
    sidecar=path.with_name(path.name+".json");sidecar.write_text(json.dumps(manifest))
    with module.NativeSocialPolicy(path,memory=False) as policy:
        candidates=[[[0.0]*width for _ in range(n)] for n in (3,2)]
        choices,_,_=policy.decide_features([41,42],[[0.0]*64]*2,candidates,[[0,0,1],[0,1]])
        assert choices==[2,1]
        memory=policy.memory_snapshot();assert set(memory["hidden"])=={"41","42"}
        with pytest.raises(ValueError):policy.decide_features([41,42],[[0.0]*64]*2,candidates,[[0,0,1],[0,0]])
        assert policy.memory_snapshot()==memory
        invalid=deepcopy(memory);invalid["hidden"]["42"][0]=2.0
        with pytest.raises(ValueError):policy.restore_memory(invalid)
        assert policy.memory_snapshot()==memory
        with pytest.raises(ValueError):policy.decide_features([41,41],[[0.0]*64]*2,candidates,[[0,0,1],[0,1]])
    manifest["interface_sources"]["training/proposals.py"]="0"*64
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="interface differs"):module.NativeSocialPolicy(path)
