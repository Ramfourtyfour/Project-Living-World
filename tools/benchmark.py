"""Measured end-to-end simulation, observations and heuristic inference throughput."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.native import World
from training.policies import HeuristicPolicy

def run(population,ticks):
    policy=HeuristicPolicy();latencies=[];tracemalloc.start();started=time.perf_counter()
    with World(42,population) as w:
        for _ in range(ticks):
            start=time.perf_counter();w.step(policy.actions(w.observations()));latencies.append(time.perf_counter()-start)
        state=w.debug_state();size=len(w.snapshot().encode("utf-8"))
    elapsed=time.perf_counter()-started;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return {"population":population,"ticks":ticks,"seconds":elapsed,"world_ticks_per_second":ticks/elapsed,
            "agent_decisions_per_second":population*ticks/elapsed,"p95_tick_ms":sorted(latencies)[int(len(latencies)*.95)]*1000,
            "snapshot_bytes":size,"python_peak_bytes":peak,"python_retained_bytes":current,
            "alive":sum(p["alive"] for p in state["people"]),"conserved":state["conserved"],
            "notes":"Includes Python JSON observation decoding, policies and native authority. tracemalloc excludes native memory; not a client FPS benchmark."}

def main():
    p=argparse.ArgumentParser();p.add_argument("--ticks",type=int,default=512);p.add_argument("--output",default="benchmarks/throughput.json");a=p.parse_args()
    report={"platform":platform.platform(),"python":platform.python_version(),"runs":[run(n,a.ticks) for n in (16,32,64)]}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
