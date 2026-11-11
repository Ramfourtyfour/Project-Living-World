"""Seed-level paired uncertainty and gate audit for the frozen v3 candidate."""
import argparse
import json
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="benchmarks/curriculum-v3-summary.json");args=p.parse_args()
    read=lambda name:json.loads((ROOT/"benchmarks"/name).read_text(encoding="utf-8"))
    validation=read("curriculum-v3-validation.json");sealed=read("curriculum-v3-sealed-test.json")
    conformance=read("native-inference-conformance.json");discovery=read("sealed-20000-native-discovery.json")
    baseline=sealed["scenarios"]["1"]["heuristic"];learned=sealed["scenarios"]["1"]["learned"]
    paired=[]
    for a,b in zip(baseline["episodes"],learned["episodes"]):
        if a["seed"]!=b["seed"]:raise ValueError("Evaluation is not seed-paired")
        paired.append({"seed":a["seed"],"heuristic_survival":a["survival_fraction"],"learned_survival":b["survival_fraction"],
                       "difference":b["survival_fraction"]-a["survival_fraction"]})
    differences=[r["difference"] for r in paired];rng=random.Random(1700)
    means=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
    conserved=all(e["conserved"] for scenario in sealed["scenarios"].values() for policy in scenario.values() for e in policy["episodes"])
    same_candidate=sealed["source_policy_sha256"]==validation["policy_sha256"] and all(r["source_policy_sha256"]==validation["policy_sha256"] for r in conformance["rows"])
    report={"version":1,"validation_gate_passed":validation["expansion_gate_passed"],"sealed_gate_passed":sealed["expansion_gate_passed"],
            "native_conformance_passed":conformance["passed"],"source_candidate_unchanged":same_candidate,"all_sealed_resources_conserved":conserved,
            "sealed_seed_count":len(paired),"paired_survival_gain":statistics.mean(differences),"paired_seed_bootstrap_95_percent_interval":[means[250],means[9749]],
            "seed_wins":sum(d>0 for d in differences),"seed_ties":sum(d==0 for d in differences),"seed_losses":sum(d<0 for d in differences),
            "paired_seeds":paired,"causal_new_supply_units":discovery["units_from_new_survival_supplies"],
            "causal_new_supply_users":discovery["individuals_using_new_supplies"],
            "pilot_gate_passed":validation["expansion_gate_passed"] and sealed["expansion_gate_passed"] and conformance["passed"] and same_candidate and conserved and discovery["units_from_new_survival_supplies"]>0,
            "limits":"Uncertainty resamples 16 whole-world seeds, not correlated individuals. Two small scene distributions and one fixed causal trace; no social learning, society, 3D or co-op claim."}
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="paired_seeds"},indent=2))

if __name__=="__main__":main()
