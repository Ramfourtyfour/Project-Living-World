"""Local native social inference timings; fixed observations, evolving own memory."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.social_loading import learned_factory,proposal_version
from training.native import World,native_fingerprint


def timing(function,repeats):
    for _ in range(8):function()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();function();samples.append((time.perf_counter_ns()-start)/1e6)
    return {"median_ms":statistics.median(samples),"p95_ms":sorted(samples)[math.ceil(0.95*len(samples))-1],"repetitions":repeats}


def main():
    p=argparse.ArgumentParser();p.add_argument("policy");p.add_argument("--output",required=True);p.add_argument("--repeats",type=int,default=100)
    args=p.parse_args();path=Path(args.output)
    if path.exists():raise FileExistsError("Preserve earlier benchmarks")
    if not 20<=args.repeats<=10000:raise ValueError("Choose 20..10000 repetitions")
    version=proposal_version(args.policy,True)
    encoder=importlib.import_module("training.proposals"+("_v2" if version==2 else ""))
    factory=learned_factory(args.policy,native=True);rows=[]
    for population in (8,32,64):
        started=time.perf_counter()
        with factory() as policy,World(40000,population) as world:
            load_ms=(time.perf_counter()-started)*1000
            for _ in range(32):world.step(policy.actions(world.observations()))
            observations={i:o for i,o in world.observations().items() if o["self"]["alive"]};ids=list(observations)
            slates=[encoder.proposals(observations[i]) for i in ids]
            x=[encoder.context(observations[i]) for i in ids];c=[[p.features for p in slate] for slate in slates];m=[[p.enabled for p in slate] for slate in slates]
            kernel=timing(lambda:policy.decide_features(ids,x,c,m),args.repeats)
            end_to_end=timing(lambda:policy.actions(observations),args.repeats)
            tracemalloc.start();policy.actions(observations);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            rows.append({"population":population,"active":len(ids),"candidates":sum(map(len,slates)),"model_and_world_load_ms":load_ms,
                         "packing_and_native_batch":kernel,"encoding_and_native_batch":end_to_end,"python_only_call_peak_bytes":peak,
                         "hidden_float_payload_bytes":64*4*len(policy.memory_snapshot()["hidden"]),"conserved":world.debug_state()["conserved"]})
            runtime=policy.runtime_sha256;model=policy.manifest["sha256"]
            print(json.dumps(rows[-1]),flush=True)
    report={"version":1,"platform":platform.platform(),"python":platform.python_version(),"processor":platform.processor(),
            "policy_sha256":model,"runtime_sha256":runtime,"world_sha256":native_fingerprint(),"proposal_version":version,"batches":rows,
            "scope":"Local fixed-observation latency with evolving per-person hidden memory; no rendering/network work; other processes may contend for CPU",
            "memory_scope":"Python allocator peak and float payload only; excludes native map/allocator/DLL overhead; not a total per-agent memory measurement"}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
