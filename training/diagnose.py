"""Development-seed behavior accounting, separate from policy scoring/test seeds."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .env import ParallelWorldEnv
from .policies import HeuristicPolicy, INTENTS, nearest

def diagnose(policy,seed,population=32,ticks=512,scenario=1):
    counts=Counter();waste=[0,0];used=[0,0];gathered=[0,0];deaths=[];invalid=Counter()
    unknown=[0,0];seen={};life=Counter();health_loss=Counter()
    with ParallelWorldEnv(population,ticks,scenario) as env:
        observations,_=env.reset(seed=seed)
        while env.agents:
            current={i:observations[i] for i in env.agents}
            intents=policy.intents(current) if hasattr(policy,"intents") else {i:policy.intent(o) for i,o in current.items()}
            for i,intent in intents.items():
                counts[INTENTS[intent]]+=1;life[i]+=1
                if intent in (0,1) and nearest(current[i],intent) is None:unknown[intent]+=1
                seen.setdefault(i,set()).update(n["id"] for n in current[i]["nodes"] if n["resource"]<2)
            next_obs,_,done,_,infos=env.step(intents)
            for i,o in next_obs.items():
                old=current[i]["self"];a=o["self"];intent=intents.get(i)
                if intent in (2,3) and not infos[i]["invalid"]:
                    r=intent-2;need=old["hunger" if r==0 else "thirst"];reduction=45 if r==0 else 55
                    waste[r]+=max(0,reduction-need);used[r]+=1
                for r in (0,1):gathered[r]+=max(0,a["inventory"][r]-old["inventory"][r])
                if infos[i]["invalid"]:invalid[str(intent)]+=1
                if a["health"]<old["health"]:
                    for name in ("hunger","thirst","fatigue"):
                        if a[name]==100:health_loss[name]+=1
                if done[i]:deaths.append({"id":i,"tick":o["tick"],"needs":[a[k] for k in ("hunger","thirst","fatigue")],"inventory":a["inventory"]})
            observations=next_obs
        return {"seed":seed,"scenario":scenario,"population":population,"ticks":ticks,
                "intents":dict(counts),"consumed_units":used,"wasted_need_reduction":waste,
                "mean_waste_per_consumed_unit":[waste[r]/max(1,used[r]) for r in (0,1)],
                "gathered":gathered,"seeking_without_known_source":unknown,"invalid":dict(invalid),
                "health_damage_ticks_by_need":dict(health_loss),"deaths":deaths,
                "alive":population-len(deaths),"mean_lifetime":statistics.mean(life.values()),
                "mean_food_water_nodes_seen":statistics.mean(map(len,seen.values()))}

def main():
    p=argparse.ArgumentParser();p.add_argument("--policy",default="inference/artifacts/curriculum-v2.pt")
    p.add_argument("--output",default="benchmarks/scarcity-diagnosis.json");args=p.parse_args()
    import torch
    from .learn import LearnedPolicy
    torch.set_num_threads(1)
    rows={}
    for name,factory in {"heuristic":HeuristicPolicy,"learned":lambda:LearnedPolicy(args.policy)}.items():
        rows[name]=[diagnose(factory(),seed) for seed in range(9000,9004)]
        print(name,json.dumps({"mean_alive":statistics.mean(r["alive"] for r in rows[name]),
                               "mean_food_waste":statistics.mean(r["mean_waste_per_consumed_unit"][0] for r in rows[name]),
                               "mean_water_waste":statistics.mean(r["mean_waste_per_consumed_unit"][1] for r in rows[name]),
                               "mean_nodes_seen":statistics.mean(r["mean_food_water_nodes_seen"] for r in rows[name])}),flush=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"version":1,"partition":"development 9000..9003; not final test","policies":rows},indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
