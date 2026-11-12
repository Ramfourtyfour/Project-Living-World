"""Rules-3 development behavior report; no sealed learning/generalization claim."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.evaluate import episode
from training.native import native_fingerprint
from training.policies import HeuristicPolicy
from inference.runtime import NativeLearnedPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="benchmarks/perception-development.json")
    p.add_argument("--overwrite",action="store_true");args=p.parse_args();path=Path(args.output)
    if path.exists() and not args.overwrite:raise SystemExit("Choose a new report or explicitly --overwrite")
    report={"version":1,"rules_version":3,"native_sha256":native_fingerprint(),"seeds":[42,43,44,45],"population":32,"ticks":512,
            "scope":"Development smoke test across daylight, sunset and night; frozen old survival brain, no new sealed claim","scenarios":{}}
    for scenario in (0,1):
        report["scenarios"][str(scenario)]={}
        for name,factory in (("heuristic",HeuristicPolicy),("frozen_native",lambda:NativeLearnedPolicy("inference/artifacts/curriculum-v3.lwgru"))):
            rows=[]
            for seed in report["seeds"]:
                policy=factory()
                try:
                    rows.append(episode(policy,seed,32,512,scenario,rules_version=3))
                    if name=="frozen_native":
                        report["policy_sha256"]=policy.manifest["sha256"];report["policy_runtime_sha256"]=policy.runtime_sha256
                finally:
                    if hasattr(policy,"close"):policy.close()
            result={"episodes":rows,"mean_survival":statistics.mean(r["survival_fraction"] for r in rows),
                    "mean_explored_cells":statistics.mean(r["mean_explored_cells"] for r in rows),"all_conserved":all(r["conserved"] for r in rows)}
            report["scenarios"][str(scenario)][name]=result
            print(json.dumps({"scenario":scenario,"policy":name,**{k:v for k,v in result.items() if k!="episodes"}}),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
