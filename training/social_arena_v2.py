"""Versioned correction: actual simultaneous reciprocal need, independent resources."""
import random
from .social_arena import Arena as LegacyArena,TASKS,replay_trace
from .native import Action
from .policies import scan


class Arena(LegacyArena):
    def _setup_step(self,actor,action):
        observed=self.world.observations();actions=self.background.actions(observed)
        actions[self.actor]=Action("rest");actions[self.partner]=Action("rest");actions[actor]=action
        _,results=self._advance(actions,"setup")
        result=next(r for r in results if r["actor"]==actor)
        # A first collision with an unseen obstacle is information, not a broken
        # experiment. The next local steering step turns/replans from that event.
        if not result["ok"] and not (action.kind=="move" and result["reason"]=="blocked"):
            raise RuntimeError(f"Setup action failed: {action}: {result}")

    def _record(self,value):
        if value.get("kind")=="header":value={**value,"scenario_version":2}
        super()._record(value)

    def _setup(self):
        self.wanted=random.Random(self.seed^0x5e71a9).randrange(2)
        if self.task!="help_reciprocal":return super()._setup()
        a,p,w=self.actor,self.partner,self.wanted
        self._acquire(a,1-w,4);source=self._acquire(p,w,6)
        # Each participant holds the other resource and lacks its own needed one.
        # Maintain only the stocked need, so both requests reflect real shortages.
        for _ in range(120):
            observed=self.world.observations()
            if observed[a]["self"]["hunger" if w==0 else "thirst"]>=65 and observed[p]["self"]["hunger" if w==1 else "thirst"]>=65:break
            actions=self.background.actions(observed)
            for actor,critical in ((a,w),(p,1-w)):
                own=observed[actor]["self"];other=1-critical;need=own["hunger" if other==0 else "thirst"]
                actions[actor]=Action("consume",resource=other) if need>=45 and own["inventory"][other] else Action("rest")
                if actor==p and actions[actor].kind=="rest" and sum(own["inventory"])<22:
                    node=next((n for n in observed[actor]["nodes"] if n["id"]==source["id"]),None)
                    if node and node["amount"]:actions[actor]=Action("gather",target=node["id"],quantity=min(2,node["amount"],22-sum(own["inventory"])))
            _,results=self._advance(actions,"setup")
            if any(not r["ok"] for r in results if r["actor"] in (a,p)):raise RuntimeError("Reciprocal setup upkeep failed")
        else:raise RuntimeError("Reciprocal needs did not arise")
        self._approach(a,self.world.observe(p)["self"])
        body=self.world.observe(a)["self"];self._setup_step(p,Action("turn",x=body["x"],y=body["y"]))
        self._setup_step(p,Action("request",target=a,resource=1-w,quantity=2))
        if not all(self.world.observe(i)["self"]["alive"] for i in (a,p)):raise RuntimeError("Reciprocal setup killed a participant")

    def report(self):return {**super().report(),"scenario_version":2}

    def _partner_action(self,o):
        action=super()._partner_action(o)
        if self.task=="help_reciprocal" and self.partner_kind=="reciprocal" and action.kind in ("move","turn","request","gather","rest"):
            relation=next((r for r in o["relationships"] if r["id"]==self.actor),{})
            if relation.get("helped",0)>0 and o["self"]["inventory"][self.wanted]>=2 and not any(p["id"]==self.actor for p in o["people"]):return scan(o)
        if self.task=="help_reciprocal" and self.partner_kind=="reciprocal" and action.kind in ("move","turn","request","gather"):
            waiting=any(r["from"]==o["self"]["id"] and r["status"] in (0,1) and o["tick"]-r["created_tick"]<=12 for r in o["requests"])
            if waiting and o["self"]["health"]>50:return Action("rest")
        return action
