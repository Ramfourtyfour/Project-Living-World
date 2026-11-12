"""Declared C2 teachers/baselines, separate from learned execution and world rules."""
import random
from .native import Action
from .policies import facing_target,move_toward
from .social import SocialHeuristicPolicy
from .proposals import proposals


class ProposalHeuristicPolicy(SocialHeuristicPolicy):
    name="local-proposal-heuristic-v1"

    def action(self,o):
        a=o["self"];inventory=a["inventory"]
        if a["hunger"]>=75 and inventory[0]:return Action("consume",resource=0)
        if a["thirst"]>=75 and inventory[1]:return Action("consume",resource=1)
        for offer in o["offers"]:
            if offer["to"]!=a["id"] or not offer["group"]:continue
            sender=next((p for p in o["people"] if p["id"]==offer["from"]),None)
            if not sender:continue
            known=[c for c in [*o["caches"],*o["place_memory"]] if c["group"]==offer["group"]]
            old=[m for m in o["place_memory"] if m["group"]==a["group"] and m["inventory_at_observation"] is not None]
            reserve=sum(sum(m["inventory_at_observation"][:2]) for m in old)
            choose=bool(known) and reserve<2 and max(a["hunger"],a["thirst"])>=45 and min(inventory[:2])<2
            return Action("accept" if choose else "decline",target=offer["id"])
        if min(inventory[:2])<2:
            current={c["id"]:c for c in o["caches"]}
            for memory in sorted(o["place_memory"],key=lambda c:(c["x"]-a["x"])**2+(c["y"]-a["y"])**2):
                if memory["group"]!=a["group"]:continue
                cache=current.get(memory["id"])
                if cache and "inventory" in cache:
                    for resource in (0,1):
                        if inventory[resource]<2 and cache["inventory"][resource]:
                            return Action("withdraw",target=cache["id"],resource=resource,quantity=min(2-inventory[resource],cache["inventory"][resource]))
                elif memory["inventory_at_observation"] is None or any(memory["inventory_at_observation"][:2]):
                    if not facing_target(o,memory["x"],memory["y"]):return Action("turn",x=memory["x"],y=memory["y"])
                    return move_toward(o,memory["x"],memory["y"])
        return super().action(o)


class RandomProposalPolicy:
    name="random-proposal-v1"
    def __init__(self,seed=0):self.rng=random.Random(seed)
    def actions(self,observations):
        return {i:self.rng.choice([p for p in proposals(o) if p.enabled]).action for i,o in observations.items() if o["self"]["alive"]}
