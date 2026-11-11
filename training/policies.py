"""Decentralized baselines and shared intent execution using local observations only."""
import math
import random
from .native import Action

INTENTS = ("forage_food", "fetch_water", "eat", "drink", "rest", "explore", "gather_wood", "wait")

def nearest(observation, resource):
    own = observation["self"]
    candidates = {n["id"]: n for n in observation["memory"] if n["resource"] == resource and n["amount_at_observation"] > 0}
    for n in observation["nodes"]:
        if n["resource"] == resource:
            if n["amount"] > 0:
                candidates[n["id"]] = n
            else:
                candidates.pop(n["id"], None)
    return min(candidates.values(), key=lambda n: ((n["x"]-own["x"])**2+(n["y"]-own["y"])**2, n["id"]), default=None)

def execute_intent(observation, intent):
    own = observation["self"]
    if intent in (0, 1, 6):
        resource = {0: 0, 1: 1, 6: 2}[intent]
        target = nearest(observation, resource)
        if target:
            if (target["x"]-own["x"])**2+(target["y"]-own["y"])**2 <= 300**2:
                # A remembered target must be seen again before interaction.
                visible = next((n for n in observation["nodes"] if n["id"] == target["id"]), None)
                if visible and visible["amount"]:
                    return Action("gather", target=target["id"], quantity=1)
            return Action("move", x=target["x"], y=target["y"])
        intent = 5
    if intent in (2, 3):
        return Action("consume", resource=intent-2)
    if intent == 4:
        return Action("rest")
    if intent == 5:
        # A stable per-person sweep, without global resource/map queries.
        angle = ((own["style"] * 0.618 + observation["tick"] // 16) % 16) * math.tau / 16
        return Action("move", x=max(0,min(24000,own["x"]+round(math.cos(angle)*800))),
                      y=max(0,min(24000,own["y"]+round(math.sin(angle)*800))))
    return Action()

class HeuristicPolicy:
    name = "heuristic-v1"

    def intent(self, observation):
        a = observation["self"]
        if a["thirst"] >= 45 and a["inventory"][1]:
            return 3
        if a["hunger"] >= 40 and a["inventory"][0]:
            return 2
        if a["fatigue"] >= 50:
            return 4
        if a["inventory"][1] < 2:
            return 1
        if a["inventory"][0] < 2:
            return 0
        return 5

    def actions(self, observations):
        return {i: execute_intent(o, self.intent(o)) for i,o in observations.items() if o["self"]["alive"]}

class RandomPolicy:
    name = "random-v1"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def actions(self, observations):
        return {i: execute_intent(o, self.rng.randrange(8)) for i,o in observations.items() if o["self"]["alive"]}

def features(observation):
    """22 actor-local scalars. No world IDs, group IDs, seed, or oracle input."""
    a = observation["self"]
    f = [a["hunger"]/100, a["thirst"]/100, a["fatigue"]/100, a["health"]/100,
         *[min(v,8)/8 for v in a["inventory"]], a["x"]/24000, a["y"]/24000, a["style"]/1000]
    for r in (0,1,2):
        n = nearest(observation,r)
        f += [float(n is not None), (n["x"]-a["x"])/24000 if n else 0,
              (n["y"]-a["y"])/24000 if n else 0]
    f.append(min(len(observation["people"]),16)/16)
    return f

def intent_mask(observation):
    a = observation["self"]
    room = sum(a["inventory"]) < 24
    return [room, room, a["inventory"][0]>0, a["inventory"][1]>0, True, True, room, True]

def features_v2(observation):
    """38 local scalars: preserve v1, then encode useful consumption and local pressure.

    Remembered resource stock/age is a belief, never a registry query. Crowding uses
    visible bodies only. These features change neither available actions nor rules.
    """
    a=observation["self"]
    f=features(observation)
    f += [float(a["inventory"][0]>0),float(a["inventory"][1]>0),
          max(0,45-a["hunger"])/45,max(0,55-a["thirst"])/55]
    visible={n["id"]:n for n in observation["nodes"]}
    for r,need in ((0,"hunger"),(1,"thirst")):
        n=nearest(observation,r)
        stock=n.get("amount",n.get("amount_at_observation",0)) if n else 0
        distance=math.hypot(n["x"]-a["x"],n["y"]-a["y"])/1800 if n else 3
        crowd=sum((p["x"]-n["x"])**2+(p["y"]-n["y"])**2<=300**2 for p in observation["people"]) if n else 0
        age=0 if n and n["id"] in visible else n.get("age",128) if n else 128
        supply=sum(node["amount"] for node in observation["nodes"] if node["resource"]==r)
        f += [min(1,stock/40),min(3,distance),min(1,crowd/16),min(1,age/128),
              min(2,supply/80),float(a[need]>=85)]
    return f
