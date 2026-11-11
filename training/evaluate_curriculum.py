"""Execute the predeclared normal+scarce pilot protocol without changing its gate."""
import argparse
import json
from pathlib import Path
import statistics

from .evaluate import episode
from .policies import HeuristicPolicy,RandomPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument("--policy",default="inference/artifacts/curriculum-v2.pt")
    p.add_argument("--native-policy",help="Evaluate the same candidate using dependency-free C++ inference")
    p.add_argument("--output",default="benchmarks/curriculum-v2-evaluation.json")
    p.add_argument("--first-seed",type=int,default=10000);p.add_argument("--seeds",type=int,default=8)
    p.add_argument("--overwrite",action="store_true");args=p.parse_args()
    path=Path(args.output)
    if path.exists() and not args.overwrite:
        raise SystemExit("Evaluation already exists; choose a new --output or explicitly --overwrite")
    if args.native_policy:
        from inference.runtime import NativeLearnedPolicy
        factory=lambda:NativeLearnedPolicy(args.native_policy)
        runtime="native"
    else:
        import torch
        from .learn import LearnedPolicy
        torch.set_num_threads(1)
        factory=lambda:LearnedPolicy(args.policy)
        runtime="pytorch"
    result={"version":1,"protocol":"paired normal/scarce comparison","seeds":list(range(args.first_seed,args.first_seed+args.seeds)),
            "ticks":512,"population":32,"scenarios":{},"runtime":runtime}
    for scenario in (0,1):
        policies={}
        for name,make_policy in {"random":lambda:RandomPolicy(17),"heuristic":HeuristicPolicy,"learned":factory}.items():
            rows=[episode(make_policy(),seed,32,512,scenario) for seed in result["seeds"]]
            summary={"episodes":rows,"mean_survival":statistics.mean(r["survival_fraction"] for r in rows),
                     "mean_lifetime":statistics.mean(r["mean_lifetime"] for r in rows),
                     "mean_explored_cells":statistics.mean(r["mean_explored_cells"] for r in rows)}
            policies[name]=summary
            print(json.dumps({"scenario":scenario,"policy":name,**{k:v for k,v in summary.items() if k!="episodes"}}),flush=True)
        result["scenarios"][str(scenario)]=policies
    normal=result["scenarios"]["0"];scarce=result["scenarios"]["1"]
    result["expansion_gate_passed"]=(normal["learned"]["mean_survival"]>=normal["heuristic"]["mean_survival"] and
                                      scarce["learned"]["mean_survival"]>scarce["heuristic"]["mean_survival"] and
                                      scarce["learned"]["mean_explored_cells"]>scarce["heuristic"]["mean_explored_cells"])
    candidate=factory();result["policy_sha256"]=candidate.manifest["sha256"]
    if args.native_policy:result["source_policy_sha256"]=candidate.manifest["source_policy_sha256"]
    if hasattr(candidate,"close"):candidate.close()
    result["claim"]="Small predeclared local pilot; no social learning or emergent society claim."
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Expansion gate:",result["expansion_gate_passed"],flush=True)

if __name__=="__main__":main()
