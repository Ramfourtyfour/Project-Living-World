"""Execute the predeclared normal+scarce pilot protocol without changing its gate."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from .evaluate import episode
from .learn import LearnedPolicy
from .policies import HeuristicPolicy,RandomPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument("--policy",default="inference/artifacts/curriculum-v2.pt")
    p.add_argument("--output",default="benchmarks/curriculum-v2-evaluation.json");args=p.parse_args()
    torch.set_num_threads(1)
    result={"version":1,"protocol":"curriculum-v2-protocol.md","seeds":list(range(10000,10008)),
            "ticks":512,"population":32,"scenarios":{}}
    for scenario in (0,1):
        policies={}
        for name,factory in {"random":lambda:RandomPolicy(17),"heuristic":HeuristicPolicy,"learned":lambda:LearnedPolicy(args.policy)}.items():
            rows=[episode(factory(),seed,32,512,scenario) for seed in result["seeds"]]
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
    result["policy_sha256"]=LearnedPolicy(args.policy).manifest["sha256"]
    result["claim"]="Small predeclared local pilot; no social learning or emergent society claim."
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Expansion gate:",result["expansion_gate_passed"],flush=True)

if __name__=="__main__":main()
