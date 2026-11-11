"""Declared social-mechanics scenarios with real acquisition and individual choices.

This measures a heuristic baseline, not learned cooperation or emergent society.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.journal import JournalRunner
from training.native import Action,native_fingerprint
from training.policies import move_toward
from training.social import SocialHeuristicPolicy

def approach(runner,actor,target):
    for _ in range(100):
        o=runner.world.observe(actor);a=o["self"]
        if (a["x"]-target["x"])**2+(a["y"]-target["y"])**2<=300**2:return
        runner.advance({actor:move_toward(o,target["x"],target["y"])})
    raise RuntimeError("Scenario setup could not reach its visible target")

def episode(seed,stock,directory):
    path=directory/f"seed-{seed}-stock-{stock}.lwj";policy=SocialHeuristicPolicy();latencies=[];events={}
    with JournalRunner.create(path,seed,8,policy=policy) as runner:
        a,b=runner.world.ids[0],runner.world.ids[4]
        source=next(n for n in runner.world.observe(b)["nodes"] if n["resource"]==0)
        approach(runner,b,source)
        remaining=stock
        while remaining:
            amount=min(2,remaining)
            result=runner.advance({b:Action("gather",target=source["id"],quantity=amount)})[0]
            if not result["ok"]:raise RuntimeError(result)
            remaining-=amount
        approach(runner,a,runner.world.observe(b)["self"])
        runner.advance({a:Action("request",target=b,resource=0,quantity=2)})
        for _ in range(32):
            started=time.perf_counter();runner.advance_policy();latencies.append(time.perf_counter()-started)
            for e in runner.world.observe(a)["events"]:events[e["id"]]=e
            request=runner.world.observe(a)["requests"][0]
            if request["status"] in (2,3,4):break
        state=runner.world.debug_state();request=runner.world.observe(a)["requests"][0]
        result={"seed":seed,"donor_stock_acquired":stock,"request_status":request["status"],
                "gift_units":sum(e["quantity"] for e in events.values() if e["type"]=="gift"),
                "conserved":state["conserved"],"state_hash":runner.world.state_hash(),
                "ticks":state["tick"],"p95_durable_tick_ms":sorted(latencies)[int(len(latencies)*0.95)]*1000,
                "source":"Real food-node gathering; a human-equivalent request; then independent per-person heuristic decisions"}
    with JournalRunner.open(path,SocialHeuristicPolicy()) as replayed:
        if replayed.world.state_hash()!=result["state_hash"]:raise RuntimeError("Social scenario replay diverged")
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=16);p.add_argument("--output",default="benchmarks/social-mechanics.json")
    p.add_argument("--journals",default="runs/social-mechanics");a=p.parse_args()
    directory=Path(a.journals);directory.mkdir(parents=True,exist_ok=True)
    results=[episode(seed,stock,directory) for seed in range(30000,30000+a.seeds) for stock in (1,4)]
    report={"version":1,"rules_version":2,"native_sha256":native_fingerprint(),"controller":SocialHeuristicPolicy.name,"episodes":results,
            "surplus_fulfilled":sum(r["request_status"]==3 for r in results if r["donor_stock_acquired"]==4),
            "scarce_declined":sum(r["request_status"]==2 for r in results if r["donor_stock_acquired"]==1),
            "all_resources_conserved":all(r["conserved"] for r in results),
            "scope":"Social mechanics and individual heuristic decisions only; no learned social behavior, faction strategy or society claim"}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="episodes"},indent=2))

if __name__=="__main__":main()
