"""Summarize recorded learner actions and terminal needs without rerunning a policy."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def diagnose(path):
    actions=Counter();invalid=Counter();deaths=[];recent={};urgent=Counter();previous={};footer=None
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        header=json.loads(next(stream));ids=header["learner_ids"]
        for line in stream:
            r=json.loads(line)
            if r["kind"]=="step" and r["phase"]=="control":
                for i in ids:
                    o=r["observations"][str(i)];a=o["self"];command=r["actions"].get(str(i),{"kind":"wait"})
                    if previous.get(i,True) and not a["alive"]:
                        deaths.append({"actor":i,"observed_tick":o["tick"],"recent_steps":recent.get(i,[])})
                    previous[i]=a["alive"]
                    if not a["alive"]:continue
                    actions[command["kind"]]+=1
                    for resource,need in ((0,"hunger"),(1,"thirst")):
                        if a[need]>=80 and a["inventory"][resource] and not (command["kind"]=="consume" and command["resource"]==resource):urgent[need]+=1
                    if a["fatigue"]>=80 and command["kind"]!="rest":urgent["fatigue"]+=1
                    row={"tick":o["tick"],"health":a["health"],"hunger":a["hunger"],"thirst":a["thirst"],"fatigue":a["fatigue"],"inventory":a["inventory"],"action":command}
                    recent.setdefault(i,[]).append(row);recent[i]=recent[i][-16:]
                for result in r["results"]:
                    if result["actor"] in ids and not result["ok"]:invalid[result["reason"]]+=1
            elif r["kind"]=="footer":footer=r
    if footer is None:raise ValueError(f"Incomplete trace: {path}")
    for i in ids:
        if previous.get(i,True) and footer.get("final_person",{}).get("id")==i and not footer["final_person"]["alive"]:
            deaths.append({"actor":i,"observed_tick":"footer","recent_steps":recent.get(i,[])})
    return {"trace":path.name,"seed":header["seed"],"task":header["task"],"scenario_version":header.get("scenario_version",1),
            "actions":dict(actions),"invalid_reasons":dict(invalid),"urgent_need_with_alternative_action":dict(urgent),"deaths":deaths,
            "learner_survivors":footer["learner_survivors"],"learner_population":footer["learner_population"],"conserved":footer["conserved"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("traces");p.add_argument("--output",required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError("Preserve previous diagnoses")
    paths=sorted(Path(args.traces).glob("*.jsonl.gz"));rows=[diagnose(path) for path in paths]
    if not rows:raise ValueError("No traces")
    report={"version":1,"scope":"Descriptive recorded-action diagnostics, not an optimal-action oracle or a gate",
            "episodes":rows,"observed_deaths":sum(len(r["deaths"]) for r in rows)}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"episodes":len(rows),"observed_deaths":report["observed_deaths"]}))


if __name__=="__main__":main()
