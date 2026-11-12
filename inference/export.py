"""Export only the declared small GRU architectures to a checked float32 format."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch
from training.learn import LearnedPolicy

ORDER=("encoder.0.weight","encoder.0.bias","memory.weight_ih_l0","memory.weight_hh_l0",
       "memory.bias_ih_l0","memory.bias_hh_l0","actor.weight","actor.bias","critic.weight","critic.bias")

def export_policy(source,target):
    source,target=Path(source),Path(target)
    manifest=target.with_name(target.name+".json")
    if target.exists() or manifest.exists():raise FileExistsError("Native artifact already exists")
    policy=LearnedPolicy(source);state=policy.model.state_dict()
    values=np.concatenate([state[name].detach().cpu().numpy().reshape(-1) for name in ORDER]).astype("<f4")
    if not np.isfinite(values).all() or (np.abs(values)>1000).any():raise ValueError("Weights exceed the bounded native contract")
    inputs=policy.model.encoder[0].in_features;hidden=policy.model.hidden_size
    header=struct.pack("<8s5I",b"LWGRU01\0",1,inputs,hidden,8,len(values))
    payload=header+values.tobytes();target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as stream:stream.write(payload)
    metadata={"version":1,"format":"lwgru-v1","architecture":policy.model.architecture,
              "feature_version":policy.model.feature_version,"inputs":inputs,"hidden":hidden,"actions":8,
              "sha256":hashlib.sha256(payload).hexdigest(),"source_policy_sha256":policy.manifest["sha256"],
              "source_method":policy.manifest["method"],"tensor_order":ORDER,
              "numeric_contract":"little-endian IEEE float32, PyTorch GRU r/z/n equations; numerical equivalence, not cross-runtime bitwise identity"}
    with manifest.open("x",encoding="utf-8") as stream:json.dump(metadata,stream,indent=2);stream.write("\n")
    return metadata

def main():
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("target");a=p.parse_args()
    print(json.dumps(export_policy(a.source,a.target),indent=2))

if __name__=="__main__":main()
