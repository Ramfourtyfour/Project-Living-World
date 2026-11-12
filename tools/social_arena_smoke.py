"""Development-only audit of physically constructed social task episodes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.social_arena import Arena,TASKS
from training.proposal_baselines import ProposalHeuristicPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=8);p.add_argument("--first-seed",type=int,default=100)
    p.add_argument("--output",default="runs/arena-smoke.json");p.add_argument("--traces");args=p.parse_args()
    policy=ProposalHeuristicPolicy();rows=[];failures=[]
    for seed in range(args.first_seed,args.first_seed+args.seeds):
        for task in TASKS:
            try:
                path=Path(args.traces)/f"{seed}-{task}.jsonl.gz" if args.traces else None
                with Arena(seed,task,horizon=64,trace=path) as arena:
                    for _ in range(64):
                        obs=arena.world.observations();_,_,done=arena.step(policy.actions({i:obs[i] for i in arena.learner_ids if obs[i]["self"]["alive"]}))
                        if done:break
                    row=arena.report();rows.append(row)
                    print(json.dumps({k:row[k] for k in ("seed","task","setup_ticks","learner_alive","invalid_actions","membership_changes","conserved")}),flush=True)
            except RuntimeError as error:
                failures.append({"seed":seed,"task":task,"error":str(error)});print(json.dumps(failures[-1]),flush=True)
    report={"version":1,"scope":"Development setup/baseline audit; not learned behavior","episodes":rows,"setup_failures":failures}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if failures:raise SystemExit(1)


if __name__=="__main__":main()
