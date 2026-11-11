"""Reused validation worlds under rules 2; never a new sealed learning claim."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from inference.runtime import NativeLearnedPolicy
from training.evaluate import episode
from training.native import World,native_fingerprint
from training.policies import HeuristicPolicy


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--policy",default="inference/artifacts/curriculum-v3.lwgru")
    p.add_argument("--seeds",type=int,default=4)
    p.add_argument("--output",default="benchmarks/rules-v2-survival-regression.json")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args();path=Path(args.output)
    if path.exists() and not args.overwrite:raise SystemExit("Choose a new output or explicitly --overwrite")
    if not 1<=args.seeds<=8:raise SystemExit("Use only the eight already inspected validation seeds")
    with World(42,1,rules_version=2) as world:
        if world.rules_version!=2:raise SystemExit("This regression requires rules 2")
    candidate=NativeLearnedPolicy(args.policy)
    report={"version":1,"rules_version":2,"native_sha256":native_fingerprint(),
            "policy_sha256":candidate.manifest["sha256"],"policy_runtime_sha256":candidate.runtime_sha256,
            "source_policy_sha256":candidate.manifest["source_policy_sha256"],
            "scope":"Previously inspected validation seeds; compatibility regression, not a new sealed claim",
            "seeds":list(range(10000,10000+args.seeds)),"ticks":512,"population":32,"scenarios":{}}
    candidate.close()
    for scenario in (0,1):
        report["scenarios"][str(scenario)]={}
        for name,factory in (("heuristic",HeuristicPolicy),("native",lambda:NativeLearnedPolicy(args.policy))):
            rows=[]
            for seed in report["seeds"]:
                policy=factory()
                try:rows.append(episode(policy,seed,32,512,scenario,rules_version=2))
                finally:
                    if hasattr(policy,"close"):policy.close()
            result={"episodes":rows,"mean_survival":statistics.mean(r["survival_fraction"] for r in rows),
                    "mean_lifetime":statistics.mean(r["mean_lifetime"] for r in rows),
                    "mean_explored_cells":statistics.mean(r["mean_explored_cells"] for r in rows)}
            report["scenarios"][str(scenario)][name]=result
            print(json.dumps({"scenario":scenario,"policy":name,**{k:v for k,v in result.items() if k!="episodes"}}),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
