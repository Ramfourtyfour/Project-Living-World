"""Declared C2/normal evaluation with physical and frozen-policy replay evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.native import native_fingerprint
from training.social_arena import TASKS,replay_trace
from training.proposal_baselines import ProposalHeuristicPolicy,RandomProposalPolicy
from training.policies import HeuristicPolicy
from tools.social_loading import learned_factory,proposal_version,scenario


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy");g.add_argument("--native-policy");g.add_argument("--baseline",choices=["social","survival","random"])
    p.add_argument("--partition",choices=["development","validation","sealed"],default="validation")
    p.add_argument("--seeds",type=int,default=16);p.add_argument("--output",required=True);p.add_argument("--frozen-candidate")
    p.add_argument("--partner",choices=["reciprocal","selfish","older"],default="reciprocal");p.add_argument("--partner-policy")
    p.add_argument("--no-memory",action="store_true");p.add_argument("--no-relationships",action="store_true")
    p.add_argument("--population",type=int,default=8)
    p.add_argument("--scenario-version",type=int,choices=[1,2],help="Defaults to the learned model version, or 2 for baselines")
    args=p.parse_args()
    if args.policy:
        factory=learned_factory(args.policy,memory=not args.no_memory,relationships=not args.no_relationships)
    elif args.native_policy:
        factory=learned_factory(args.native_policy,native=True,memory=not args.no_memory,relationships=not args.no_relationships)
    else:factory={"social":ProposalHeuristicPolicy,"survival":HeuristicPolicy,"random":RandomProposalPolicy}[args.baseline]
    version=args.scenario_version or (proposal_version(args.policy or args.native_policy,bool(args.native_policy)) if args.policy or args.native_policy else 2)
    Arena=scenario(version)
    if args.partner=="older" and not args.partner_policy:raise SystemExit("Older partner evaluation requires --partner-policy")
    if args.partner_policy:
        partner_factory=learned_factory(args.partner_policy)
    else:partner_factory=lambda:HeuristicPolicy() if args.partner=="selfish" else None
    probe=factory();identifier=probe.name;digest=getattr(probe,"manifest",{}).get("sha256")
    if hasattr(probe,"close"):probe.close()
    if args.partition=="sealed":
        if not args.frozen_candidate or not digest:raise SystemExit("Sealed evaluation requires a frozen learned candidate")
        frozen=json.loads(Path(args.frozen_candidate).read_bytes())
        required={"policy_sha256":digest,"scenario_version":version,"population":args.population,"partner":args.partner,
                  "recurrent_memory":not args.no_memory,"relationship_features":not args.no_relationships}
        if any(frozen.get(k)!=v for k,v in required.items()) or frozen.get("validation_seeds")!=list(range(50000,50016)) or not frozen.get("validation_gate_passed"):
            raise SystemExit("Candidate/configuration has not passed the complete declared validation gate")
    limit=32 if args.partition=="sealed" else 16 if args.partition=="validation" else 64
    if not 1<=args.seeds<=limit:raise SystemExit("Seed count exceeds this declared partition")
    first={"development":40000,"validation":50000,"sealed":60000}[args.partition]
    path=Path(args.output)
    if path.exists():raise SystemExit("Preserve existing evaluations; choose a new output")
    directory=path.with_suffix("");directory.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in range(first,first+args.seeds):
        for task in TASKS:
            policy=factory();partner=partner_factory();trace=directory/f"{seed}-{task}.jsonl.gz"
            try:
                with Arena(seed,task,horizon=512 if task=="survival" else 128 if version==2 else 64,trace=trace,population=args.population,
                           partner=partner.name if partner else args.partner,
                           learner_fraction=0.5 if task=="survival" and partner else 1) as arena:
                    arena._record({"kind":"controller","id":policy.name,"policy_sha256":digest,"partner":partner.name if partner else args.partner})
                    for _ in range(arena.horizon):
                        o=arena.world.observations();actions=policy.actions({i:o[i] for i in arena.learner_ids if o[i]["self"]["alive"]})
                        _,_,done=arena.step(actions,partner)
                        if done:break
                    row=arena.report();actor=arena.actor;other=arena.partner
                    cache_groups={c["id"]:c["group"] for c in arena.world.debug_state()["caches"]}
                    accepted=bool(set(row["initial_offers"])&set(row["accepted_offers"]))
                    refused=bool(set(row["initial_offers"])&set(row["declined_offers"])) and not accepted
                    used=[u for u in row["useful_consumption"] if u["useful_need_reduction"]>=20]
                    traded=any(u["actor"]==actor and u["via"]=="trade" and u["source"]==other for u in used)
                    received=any(u["actor"]==actor and u["via"]=="gift" and u["source"]==other for u in used)
                    helped=any(u["actor"]==other and u["via"]=="gift" and u["source"]==actor for u in used)
                    joined={e["to"] for e in row["membership_changes"] if e["actor"]==actor and e["to"] and e["to"]!=e["from"]}
                    accessed=any(u["actor"]==actor and u["via"]=="cache" and cache_groups.get(u["source"]) in joined for u in used)
                    success={"trade_good":accepted and traded,"trade_bad":refused,"help_reciprocal":received and helped,
                             "invite_good":accepted and bool(joined) and accessed,"invite_bad":refused,"survival":True}[task]
                    row.update({"social_success":bool(success and row["learner_alive"]),"accepted_initial":accepted,"refused_initial":refused,
                                "useful_trade":traded,"reciprocal_useful_aid":received and helped,"useful_new_agreement_access":accessed,
                                "learner_survival":row["learner_survivors"]/row["learner_population"]})
            finally:
                if hasattr(policy,"close"):policy.close()
                if hasattr(partner,"close"):partner.close()
            verifier=factory()
            try:verified=replay_trace(trace,verifier if digest else None)
            finally:
                if hasattr(verifier,"close"):verifier.close()
            row["replayed"]=verified["state_hash"]==row["state_hash"];row["trace_sha256"]=hashlib.sha256(trace.read_bytes()).hexdigest();rows.append(row)
            print(json.dumps({"seed":seed,"task":task,"success":row["social_success"],"survival":row["learner_survival"],"replayed":row["replayed"]}),flush=True)
    summary={task:{"episodes":len([r for r in rows if r["task"]==task]),"success_rate":statistics.mean(r["social_success"] for r in rows if r["task"]==task),
                   "mean_learner_survival":statistics.mean(r["learner_survival"] for r in rows if r["task"]==task)} for task in TASKS}
    thresholds=all(summary[t]["success_rate"]>=0.75 for t in TASKS if t!="survival") and summary["survival"]["mean_learner_survival"]>=0.95 and all(r["conserved"] and r["replayed"] for r in rows)
    gate=thresholds and args.seeds>=(32 if args.partition=="sealed" else 16)
    report={"version":2,"scenario_version":version,"controller":identifier,"policy_sha256":digest,"native_sha256":native_fingerprint(),"partition":args.partition,"seeds":list(range(first,first+args.seeds)),
            "partner":args.partner,"population":args.population,"recurrent_memory":not args.no_memory,"relationship_features":not args.no_relationships,
            "episodes":rows,"summary":summary,"diagnostic_thresholds_met":thresholds,"bounded_c2_gate_passed":gate,"full_lw008_gate_passed":False,
            "scope":"Bounded physically constructed C2 tasks plus normal survival; whole-world society/generalization is a separate requirement"}
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"summary":summary,"bounded_c2_gate_passed":gate},indent=2))


if __name__=="__main__":main()
