"""Verify a private journal and measure actual use of newly discovered supplies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.journal import JournalRunner
from training.policies import HeuristicPolicy

class DiscoveryTrace:
    def __init__(self):
        self.initial={};self.seen={};self.new_seen={};self.harvested={};self.units=Counter();self.first={}

    def __call__(self,payload,observations):
        if payload["kind"]=="header":
            for i,o in observations.items():
                self.initial[i]={n["id"] for n in o["nodes"] if n["resource"]<2}
                self.seen[i]=set(self.initial[i]);self.new_seen[i]=set();self.harvested[i]=set();self.units[i]=0
            return
        accepted={r["actor"] for r in payload["results"] if r["ok"]}
        for i,o in observations.items():
            visible={n["id"] for n in o["nodes"] if n["resource"]<2}
            discovered=visible-self.seen[i]
            if discovered and i not in self.first:self.first[i]=o["tick"]
            self.new_seen[i].update(discovered);self.seen[i].update(visible)
        for command in payload["actions"]:
            i=command["actor"];action=command["action"]
            if i not in accepted or action["kind"]!="gather" or action["target"] not in self.new_seen[i]:continue
            node=next((n for n in observations[i]["nodes"] if n["id"]==action["target"] and n["resource"]<2),None)
            if node is not None:
                self.harvested[i].add(node["id"]);self.units[i]+=action["quantity"]

    def report(self):
        return {"mean_new_supply_nodes_seen":statistics.mean(map(len,self.new_seen.values())),
                "mean_new_supply_nodes_harvested":statistics.mean(map(len,self.harvested.values())),
                "individuals_using_new_supplies":sum(bool(v) for v in self.harvested.values()),
                "units_from_new_survival_supplies":sum(self.units.values()),
                "individuals":[{"id":i,"new_supplies_seen":sorted(self.new_seen[i]),"new_supplies_harvested":sorted(self.harvested[i]),
                                "new_supply_units":self.units[i],"first_new_supply_tick":self.first.get(i)} for i in sorted(self.initial)]}

def main():
    p=argparse.ArgumentParser();p.add_argument("journal");p.add_argument("--native-policy");p.add_argument("--output",required=True);a=p.parse_args()
    policy=HeuristicPolicy()
    if a.native_policy:
        from inference.runtime import NativeLearnedPolicy
        policy=NativeLearnedPolicy(a.native_policy)
    trace=DiscoveryTrace()
    with JournalRunner.open(a.journal,policy,trace) as runner:
        report={"version":1,"controller":policy.name,"final_state_hash":runner.world.state_hash(),
                "final_tick":runner.world.observe(runner.world.ids[0])["tick"],"conserved":runner.world.debug_state()["conserved"],**trace.report(),
                "scope":"Actual accepted food/water harvests from sources absent from that individual's initial sight. A causal trace, not proof of social learning."}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="individuals"},indent=2))

if __name__=="__main__":main()
