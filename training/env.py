"""PettingZoo-style simultaneous API; intentionally no PettingZoo dependency/subclass."""
from .native import World
from .policies import execute_intent

class ParallelWorldEnv:
    metadata = {"name": "living_world_v1", "is_parallelizable": True}

    def __init__(self, population=32, max_cycles=512, scenario=0, reward_version=1):
        self.population, self.max_cycles = population, max_cycles
        self.scenario=scenario
        if reward_version not in (1,2):
            raise ValueError("Unsupported reward version")
        self.reward_version=reward_version
        self.world = None
        self.agents = []
        self.possible_agents = []
        self.visited = {}

    def reset(self, seed=1, options=None):
        self.close()
        self.world = World(seed, self.population, scenario=self.scenario)
        self.agents = list(self.world.ids)
        self.possible_agents = list(self.agents)
        observations = self.world.observations()
        self.visited = {i: {(o["self"]["x"]//1000,o["self"]["y"]//1000)} for i,o in observations.items()}
        return observations, {i:{} for i in self.agents}

    def step(self, actions):
        if self.world is None:
            raise RuntimeError("Call reset before step")
        if set(actions) - set(self.agents):
            raise ValueError("Actions include inactive or unknown actors")
        before = {i: self.world.observe(i) for i in self.agents}
        native_actions = {i: execute_intent(before[i],a) if isinstance(a,int) else a for i,a in actions.items()}
        results = self.world.step(native_actions, requests={i:o["self"]["last_request"]+1 for i,o in before.items()})
        failures = {r["actor"] for r in results if not r["ok"]}
        observations, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        for i in self.agents:
            o = self.world.observe(i); a = o["self"]; old = before[i]["self"]
            cell = (a["x"]//1000,a["y"]//1000)
            discovery = cell not in self.visited[i]
            self.visited[i].add(cell)
            components = {
                "survival": 0.1 if a["alive"] else -5.0,
                "health": (a["health"]-old["health"])*0.1,
                "needs": ((old["hunger"]-a["hunger"])+(old["thirst"]-a["thirst"])) * 0.012,
                "rest": (old["fatigue"]-a["fatigue"])*0.008,
                "exploration": 0.08 if discovery and a["health"]>50 and max(a["hunger"],a["thirst"])<85 else 0.0,
                "invalid": -0.1 if i in failures else 0.0,
            }
            if self.reward_version==2:
                action=native_actions.get(i)
                wasted=0
                if action and action.kind=="consume" and i not in failures and action.resource in (0,1):
                    maximum=45 if action.resource==0 else 55
                    need=old["hunger" if action.resource==0 else "thirst"]
                    wasted=max(0,maximum-need)
                components["waste"]=-0.02*wasted
            observations[i] = o; rewards[i] = sum(components.values())
            terminated[i] = not a["alive"]; truncated[i] = o["tick"]>=self.max_cycles and a["alive"]
            infos[i] = {"reward_components": components, "invalid": i in failures}
        self.agents = [i for i in self.agents if not terminated[i] and not truncated[i]]
        return observations, rewards, terminated, truncated, infos

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
