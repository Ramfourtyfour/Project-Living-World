"""Run/resume a durable authoritative local world and publish a complete checkpoint."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.checkpoint import save_checkpoint
from training.journal import JournalRunner,recover_incomplete_tail
from training.policies import HeuristicPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="runs/world.lwj")
    p.add_argument("--resume",action="store_true");p.add_argument("--verify-only",action="store_true")
    p.add_argument("--policy");p.add_argument("--seed",type=int,default=42)
    p.add_argument("--native-policy",help="Use exported C++ policy inference without PyTorch")
    p.add_argument("--population",type=int,default=32);p.add_argument("--scenario",type=int,choices=[0,1],default=0)
    p.add_argument("--ticks",type=int,default=128);p.add_argument("--checkpoint");p.add_argument("--report")
    p.add_argument("--recover-to",help="Recover only an incomplete final write into a new log; preserve the original")
    args=p.parse_args();policy=HeuristicPolicy()
    if args.policy and args.native_policy:raise SystemExit("Choose one policy runtime")
    if args.native_policy:
        from inference.runtime import NativeLearnedPolicy
        policy=NativeLearnedPolicy(args.native_policy)
    elif args.policy:
        import torch
        from training.learn import LearnedPolicy
        torch.set_num_threads(1);policy=LearnedPolicy(args.policy)
    if args.recover_to:
        print(json.dumps(recover_incomplete_tail(args.output,args.recover_to,policy),indent=2));return
    started=time.perf_counter()
    runner=JournalRunner.open(args.output,policy) if args.resume or args.verify_only else JournalRunner.create(args.output,args.seed,args.population,args.scenario,policy)
    replay_seconds=time.perf_counter()-started;latencies=[]
    with runner:
        if not args.verify_only:
            for _ in range(args.ticks):
                tick=time.perf_counter();runner.advance_policy();latencies.append(time.perf_counter()-tick)
        if args.checkpoint:save_checkpoint(runner.world,args.checkpoint,policy)
        state=runner.world.debug_state()
        report={"version":1,"tick":state["tick"],"population":len(runner.world.ids),
                "alive":sum(p["alive"] for p in state["people"]),"state_hash":runner.world.state_hash(),
                "conserved":state["conserved"],"controller":policy.name,"journal_bytes":Path(args.output).stat().st_size,
                "create_or_replay_seconds":replay_seconds,
                "durable_step_p95_ms":sorted(latencies)[min(len(latencies)-1,int(len(latencies)*0.95))]*1000 if latencies else None,
                "mean_step_ms":statistics.mean(latencies)*1000 if latencies else None,
                "notes":"Single-writer copy-before-commit with file fsync; process-interruption tests, not a hardware power-loss guarantee."}
    if args.report:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
