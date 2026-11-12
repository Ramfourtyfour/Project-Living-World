"""Export the declared social GRU/slate model into a separate native runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from training.social_model_v2 import LearnedSocialPolicy,ARCHITECTURE
from training.proposals_v2 import BASE_SIZE,CANDIDATE_SIZE,MAX_CANDIDATES,KINDS

ROOT=Path(__file__).resolve().parents[1]
ORDER=("encoder.0.weight","encoder.0.bias","memory.weight_ih_l0","memory.weight_hh_l0","memory.bias_ih_l0","memory.bias_hh_l0",
       "survival_actor.weight","survival_actor.bias","scorer.0.weight","scorer.0.bias","scorer.2.weight","scorer.2.bias","critic.weight","critic.bias")
INTERFACE=("training/proposals.py","training/proposals_v2.py","training/policies.py","training/native.py")


def export_policy(source,target):
    source,target=Path(source),Path(target);manifest=target.with_name(target.name+".json")
    if target.exists() or manifest.exists():raise FileExistsError("Preserve existing native social artifacts")
    policy=LearnedSocialPolicy(source);m=policy.manifest
    interface={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in INTERFACE}
    expected={k.replace('\\','/'):v for k,v in m.get("configuration",{}).get("source_sha256",{}).items()}
    if expected and any(expected.get(name)!=digest for name,digest in interface.items()):raise ValueError("Proposal/decoder source differs from training")
    state=policy.model.state_dict();values=np.concatenate([state[name].numpy().reshape(-1) for name in ORDER]).astype("<f4")
    if len(values)!=38026 or not np.isfinite(values).all() or (np.abs(values)>1000).any():raise ValueError("Invalid bounded social weights")
    payload=struct.pack("<8s6I",b"LWSLAT2\0",2,BASE_SIZE,CANDIDATE_SIZE,64,MAX_CANDIDATES,len(values))+values.tobytes()
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as stream:stream.write(payload)
    metadata={"version":2,"format":"lwslate-v2","architecture":ARCHITECTURE,"observation_version":3,"proposal_version":2,
              "base_inputs":BASE_SIZE,"candidate_inputs":CANDIDATE_SIZE,"hidden":64,"max_candidates":MAX_CANDIDATES,"candidate_kinds":list(KINDS),
              "sha256":hashlib.sha256(payload).hexdigest(),"source_policy_sha256":m["sha256"],"interface_sources":interface,"tensor_order":ORDER,
              "numeric_contract":"little-endian float32; standard PyTorch GRU r/z/n and shared candidate MLP; numerical, not bitwise equivalence"}
    with manifest.open("x",encoding="utf-8") as stream:json.dump(metadata,stream,indent=2);stream.write("\n")
    return metadata


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("target");args=p.parse_args()
    print(json.dumps(export_policy(args.source,args.target),indent=2))
