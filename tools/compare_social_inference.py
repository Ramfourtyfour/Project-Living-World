"""Compare trained social weights and individual memories on real trajectories."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import importlib
from training.social_arena import TASKS
from tools.social_loading import learned_factory,proposal_version,scenario


def main():
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("native");p.add_argument("--output",required=True);p.add_argument("--seeds",type=int,default=2)
    args=p.parse_args();torch.set_num_threads(1);rows=[]
    version=proposal_version(args.source)
    if proposal_version(args.native,True)!=version:raise ValueError("Different proposal versions")
    reference_factory=learned_factory(args.source);native_factory=learned_factory(args.native,native=True)
    tensors=importlib.import_module("training.social_model"+("_v2" if version==2 else "")).tensors
    Arena=scenario(version)
    for seed in range(40000,40000+args.seeds):
        for task in TASKS:
            reference=reference_factory();hidden={};decisions=0;disagreements=0;max_logit=0;max_hidden=0;max_value=0
            with native_factory() as native,Arena(seed,task,horizon=256 if task=="survival" else 128 if version==2 else 64) as arena:
                if native.manifest["source_policy_sha256"]!=reference.manifest["sha256"]:raise ValueError("Another source policy was exported")
                for _ in range(arena.horizon):
                    observed=arena.world.observations();ids=[i for i in arena.learner_ids if observed[i]["self"]["alive"]]
                    if not ids:break
                    x,c,mask,slates=tensors(observed,ids);prior=torch.stack([hidden.get(i,torch.zeros(64)) for i in ids])[None]
                    with torch.no_grad():logits,values,state=reference.model(x[None],c[None],prior)
                    expected=logits[0].masked_fill(~mask,-1e9).argmax(-1).tolist()
                    actual,scores,estimate=native.decide_features(ids,x.tolist(),[[p.features for p in slate] for slate in slates],[[p.enabled for p in slate] for slate in slates])
                    decisions+=len(ids);disagreements+=sum(a!=b for a,b in zip(actual,expected))
                    for j,slate in enumerate(slates):max_logit=max(max_logit,float((logits[0,j,:len(slate)]-torch.tensor(scores[j])).abs().max()))
                    max_value=max(max_value,float((values[0]-torch.tensor(estimate)).abs().max()))
                    hidden.update({i:state[0,j].clone() for j,i in enumerate(ids)})
                    memory=native.memory_snapshot()["hidden"]
                    for i in ids:max_hidden=max(max_hidden,float((hidden[i]-torch.tensor(memory[str(i)])).abs().max()))
                    _,_,done=arena.step({i:slates[j][expected[j]].action for j,i in enumerate(ids)})
                    if done:break
                rows.append({"seed":seed,"task":task,"decisions":decisions,"choice_disagreements":disagreements,"max_logit_error":max_logit,
                             "max_hidden_error":max_hidden,"max_value_error":max_value,"conserved":arena.report()["conserved"]})
                native_sha=native.runtime_sha256;model_sha=native.manifest["sha256"];source_sha=reference.manifest["sha256"]
                print(json.dumps(rows[-1]),flush=True)
    report={"version":2,"scenario_version":version,"scope":"Development trajectories, trained weights; shared observations, separate individual recurrent memory",
            "runtime_sha256":native_sha,"policy_sha256":model_sha,"source_policy_sha256":source_sha,"episodes":rows,
            "decisions":sum(r["decisions"] for r in rows),"choice_disagreements":sum(r["choice_disagreements"] for r in rows),
            "max_logit_error":max(r["max_logit_error"] for r in rows),"max_hidden_error":max(r["max_hidden_error"] for r in rows)}
    path=Path(args.output)
    if path.exists():raise FileExistsError("Preserve previous conformance results")
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if report["choice_disagreements"] or report["max_logit_error"]>0.0002 or report["max_hidden_error"]>0.00002:raise SystemExit("Conformance gate failed")


if __name__=="__main__":main()
