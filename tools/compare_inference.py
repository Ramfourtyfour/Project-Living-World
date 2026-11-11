"""Compare exported native GRU to its PyTorch reference on development trajectories."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from inference.runtime import NativeLearnedPolicy
from training.learn import LearnedPolicy,tensor_observations
from training.native import World
from training.policies import execute_intent

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",default="inference/artifacts/curriculum-v3.pt")
    p.add_argument("--native",default="inference/artifacts/curriculum-v3.lwgru")
    p.add_argument("--output",default="benchmarks/native-inference-conformance.json");args=p.parse_args()
    torch.set_num_threads(1);rows=[]
    for scenario,seed in ((0,9010),(1,9011)):
        reference=LearnedPolicy(args.source);hidden={};logit_error=0;memory_error=0;decisions=0;disagreements=0;native_ms=[];torch_ms=[]
        with NativeLearnedPolicy(args.native) as native,World(seed,32,scenario=scenario) as world:
            if native.manifest["source_policy_sha256"]!=reference.manifest["sha256"]:raise ValueError("Native artifact has another source policy")
            for _ in range(512):
                observations={i:o for i,o in world.observations().items() if o["self"]["alive"]}
                ids=list(observations)
                if not ids:break
                x,mask=tensor_observations(observations,ids,reference.model.feature_version)
                prior=torch.stack([hidden.get(i,torch.zeros(reference.model.hidden_size)) for i in ids]).unsqueeze(0)
                start=time.perf_counter()
                with torch.no_grad():logits,_,state=reference.model(x.unsqueeze(0),prior)
                expected=logits[0].masked_fill(~mask,-1e9).argmax(-1).tolist();torch_ms.append((time.perf_counter()-start)*1000)
                start=time.perf_counter();actual,native_logits,_=native.decide_features(ids,x.tolist(),mask.tolist());native_ms.append((time.perf_counter()-start)*1000)
                logit_error=max(logit_error,float((torch.tensor(native_logits)-logits[0]).abs().max()))
                decisions+=len(ids);disagreements+=sum(a!=b for a,b in zip(actual,expected))
                hidden.update({i:state[0,j].clone() for j,i in enumerate(ids)})
                memory=native.memory_snapshot()["hidden"]
                for i in ids:memory_error=max(memory_error,float((torch.tensor(memory[str(i)])-hidden[i]).abs().max()))
                world.step({i:execute_intent(observations[i],expected[j]) for j,i in enumerate(ids)})
            percentile=lambda samples:sorted(samples)[min(len(samples)-1,int(len(samples)*0.95))]
            row={"scenario":scenario,"development_seed":seed,"decisions":decisions,"intent_disagreements":disagreements,
                 "max_logit_absolute_error":logit_error,"max_hidden_absolute_error":memory_error,
                 "native_batch_p95_ms":percentile(native_ms),"pytorch_batch_p95_ms":percentile(torch_ms),
                 "native_runtime_sha256":native.runtime_sha256,"source_policy_sha256":reference.manifest["sha256"],
                 "native_policy_sha256":native.manifest["sha256"],"conserved":world.debug_state()["conserved"]}
            rows.append(row);print(json.dumps(row),flush=True)
    report={"version":1,"rows":rows,"passed":all(r["intent_disagreements"]==0 and r["max_logit_absolute_error"]<=1e-4 and r["max_hidden_absolute_error"]<=1e-4 for r in rows),
            "scope":"Two development trajectories with shared inputs. Numerical conformance, not bitwise equivalence or a general-purpose model runtime."}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if not report["passed"]:raise SystemExit("Native inference conformance failed")

if __name__=="__main__":main()
