"""Record every action, personal observation, result and state hash for replay."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.native import Action, World
from training.policies import HeuristicPolicy, RandomPolicy

def record(path, seed=42, population=32, ticks=256, policy_name="heuristic"):
    policy = HeuristicPolicy() if policy_name=="heuristic" else RandomPolicy(seed)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with World(seed,population) as world, path.open("w",encoding="utf-8") as out:
        out.write(json.dumps({"version":1,"policy":policy.name,"snapshot":world.snapshot()})+"\n")
        for _ in range(ticks):
            observations = world.observations()
            actions = policy.actions(observations)
            results = world.step(actions)
            out.write(json.dumps({"observations":observations,"actions":{i:asdict(a) for i,a in actions.items()},
                                  "results":results,"hash":world.state_hash()})+"\n")
        state = world.debug_state()
        world.save(path.with_suffix(".lws"))
    return {"seed":seed,"ticks":ticks,"alive":sum(p["alive"] for p in state["people"]),
            "population":population,"hash":state["hash"],"conserved":state["conserved"]}

def replay(path):
    with Path(path).open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header["version"]!=1:
            raise ValueError("Unsupported replay version")
        with World(snapshot=header["snapshot"]) as world:
            count = 0
            for line in stream:
                step = json.loads(line)
                actual = world.observations()
                expected = {int(i):o for i,o in step["observations"].items()}
                if actual!=expected:
                    raise ValueError(f"Observation divergence before tick {count+1}")
                results = world.step({int(i):Action(**a) for i,a in step["actions"].items()})
                if results!=step["results"] or world.state_hash()!=step["hash"]:
                    raise ValueError(f"State divergence at tick {count+1}")
                count += 1
            return {"verified_ticks":count,"hash":world.state_hash()}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--population",type=int,default=32);p.add_argument("--ticks",type=int,default=256)
    p.add_argument("--output",default="runs/rollout.jsonl");p.add_argument("--policy",choices=["heuristic","random"],default="heuristic")
    p.add_argument("--replay");a=p.parse_args()
    print(json.dumps(replay(a.replay) if a.replay else record(a.output,a.seed,a.population,a.ticks,a.policy),indent=2))

if __name__=="__main__":
    main()
