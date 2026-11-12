"""Paired held-out evaluation. Report distributions; never infer cooperation from reward."""
import argparse
import json
from pathlib import Path
import statistics
import time
from .env import ParallelWorldEnv
from .policies import HeuristicPolicy,RandomPolicy

def episode(policy,seed,population,ticks,scenario=0,rules_version=3):
    started=time.perf_counter()
    with ParallelWorldEnv(population,ticks,scenario,rules_version=rules_version) as env:
        observations,_=env.reset(seed=seed)
        invalid=0;steps=0;reward=0;lifetimes={i:0 for i in env.possible_agents};inference=[]
        while env.agents:
            start=time.perf_counter();actions=policy.actions({i:observations[i] for i in env.agents})
            inference.append(time.perf_counter()-start)
            observations,rewards,done,truncated,infos=env.step(actions)
            invalid+=sum(info["invalid"] for info in infos.values());steps+=len(rewards);reward+=sum(rewards.values())
            for i in rewards:lifetimes[i]+=1
        state=env.world.debug_state()
        return {"seed":seed,"alive":sum(p["alive"] for p in state["people"]),"population":population,
                "survival_fraction":sum(p["alive"] for p in state["people"])/population,
                "mean_lifetime":statistics.mean(lifetimes.values()),"mean_explored_cells":statistics.mean(map(len,env.visited.values())),
                "reward_per_agent":reward/population,"invalid_action_rate":invalid/max(1,steps),"conserved":state["conserved"],
                "final_hash":state["hash"],"seconds":time.perf_counter()-started,
                "inference_batch_p95_ms":sorted(inference)[min(len(inference)-1,int(len(inference)*0.95))]*1000}

def main():
    p=argparse.ArgumentParser();p.add_argument("--policy",default="inference/artifacts/pilot.pt")
    p.add_argument("--seeds",type=int,default=8);p.add_argument("--population",type=int,default=32)
    p.add_argument("--ticks",type=int,default=512);p.add_argument("--output",default="benchmarks/pilot-evaluation.json")
    p.add_argument("--scenario",type=int,choices=[0,1],default=0)
    p.add_argument("--rules-version",type=int,choices=[1,2,3],default=3)
    args=p.parse_args();factories={"random":lambda:RandomPolicy(17),"heuristic":HeuristicPolicy}
    if Path(args.policy).is_file():
        import torch
        from .learn import LearnedPolicy
        torch.set_num_threads(1)
        factories["learned"]=lambda:LearnedPolicy(args.policy)
    report={"version":1,"rules_version":args.rules_version,"scenario":args.scenario,"ticks":args.ticks,"population":args.population,"held_out_seeds":list(range(10000,10000+args.seeds)),"policies":{}}
    for name,factory in factories.items():
        rows=[episode(factory(),seed,args.population,args.ticks,args.scenario,args.rules_version) for seed in report["held_out_seeds"]]
        report["policies"][name]={"episodes":rows,"mean_survival":statistics.mean(r["survival_fraction"] for r in rows),
                                  "mean_explored_cells":statistics.mean(r["mean_explored_cells"] for r in rows),
                                  "mean_lifetime":statistics.mean(r["mean_lifetime"] for r in rows)}
        print(name,json.dumps({k:v for k,v in report["policies"][name].items() if k!="episodes"}),flush=True)
    baseline=report["policies"]["heuristic"];learned=report["policies"].get("learned")
    report["expansion_gate_passed"]=bool(learned and learned["mean_survival"]>baseline["mean_survival"] and learned["mean_explored_cells"]>baseline["mean_explored_cells"])
    report["claim"]="Small local pilot only. No demonstrated emergent society, social learning, or general MARL superiority."
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Expansion gate:",report["expansion_gate_passed"],flush=True)

if __name__=="__main__":main()
