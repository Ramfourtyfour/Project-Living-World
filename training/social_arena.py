"""C2 tasks constructed exclusively through real rules-3 actions.

The setup director arranges experiments, not live NPC decisions. Setup is logged,
costs world time, and never edits snapshots, inventories, needs or membership.
"""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random

from .native import Action,World,native_fingerprint
from .policies import HeuristicPolicy,move_toward,facing_target
from .social import SocialHeuristicPolicy
from .provenance import StockTrace

TASKS=("trade_good","trade_bad","help_reciprocal","invite_good","invite_bad","survival")


class Arena:
    def __init__(self,seed,task,*,population=8,horizon=64,trace=None,partner="reciprocal",learner_fraction=1):
        if task not in TASKS:raise ValueError("Unknown social curriculum task")
        self.seed,self.task,self.horizon,self.partner_kind=seed,task,horizon,partner
        self.rng=random.Random(seed);self.world=World(seed,population,scenario=0 if task=="survival" else 1)
        camp=seed%4
        self.actor=self.world.ids[camp];self.partner=self.world.ids[camp+4] if population>=8 else self.world.ids[(camp+1)%population]
        self.background=HeuristicPolicy();self.social=SocialHeuristicPolicy();self.trace=None
        self.learner_ids=sorted(self.rng.sample(list(self.world.ids),max(1,int(population*learner_fraction)))) if task=="survival" else [self.actor]
        self.visited={i:set() for i in self.world.ids};self.decisions=0;self.invalid=0;self.memberships=[];self.transfers=[]
        self.initial_group=self.world.observe(self.actor)["self"]["group"]
        if trace:
            path=Path(trace);path.parent.mkdir(parents=True,exist_ok=True)
            self.trace=gzip.open(path,"xt",encoding="utf-8",compresslevel=1)
        self._record({"kind":"header","version":1,"native_sha256":native_fingerprint(),"seed":seed,"task":task,"partner":partner,"population":population,
                      "learner_ids":self.learner_ids,"primary_actor":self.actor,"partner_id":self.partner,"snapshot":self.world.snapshot()})
        self.wanted=seed%2
        if task!="survival":self._setup()
        self.start_tick=self.world.observe(self.actor)["tick"]
        self.start_state=self.world.observe(self.actor)["self"]
        self.stock=StockTrace(self.world.debug_state());self.useful=[];self.accepted=[];self.declined=[]
        self.initial_offers=[f["id"] for f in self.world.observe(self.actor)["offers"] if f["to"]==self.actor]
        self._record({"kind":"control_start","tick":self.start_tick,"snapshot":self.world.snapshot()})

    def _record(self,value):
        if self.trace:self.trace.write(json.dumps(value,separators=(",",":"),allow_nan=False)+"\n")

    def _advance(self,actions,phase):
        observations=self.world.observations();results=self.world.step(actions)
        self._record({"kind":"step","phase":phase,"observations":observations,"actions":{str(i):asdict(a) for i,a in actions.items()},
                      "results":results,"state_hash":self.world.state_hash()})
        return observations,results

    def _safe_action(self,actor):
        o=self.world.observe(actor);a=o["self"]
        if a["thirst"]>=70 and a["inventory"][1]:return Action("consume",resource=1)
        if a["hunger"]>=70 and a["inventory"][0]:return Action("consume",resource=0)
        if a["fatigue"]>=80:return Action("rest")
        return None

    def _setup_step(self,actor,action):
        observations=self.world.observations();actions=self.background.actions(observations)
        # Experimental participants do only explicitly logged setup actions, while
        # other individuals keep their own survival baseline.
        actions[self.actor]=Action("rest");actions[self.partner]=Action("rest")
        actions[actor]=action
        _,results=self._advance(actions,"setup")
        result=next(r for r in results if r["actor"]==actor)
        if not result["ok"]:raise RuntimeError(f"Setup action failed: {action}: {result}")

    def _approach(self,actor,point,radius=240):
        for _ in range(180):
            o=self.world.observe(actor);a=o["self"]
            if (a["x"]-point["x"])**2+(a["y"]-point["y"])**2<=radius**2:
                if not facing_target(o,point["x"],point["y"]):self._setup_step(actor,Action("turn",x=point["x"],y=point["y"]))
                return
            self._setup_step(actor,self._safe_action(actor) or move_toward(o,point["x"],point["y"]))
        raise RuntimeError("Setup could not reach a waypoint")

    def _find(self,actor,resource):
        for _ in range(12):
            o=self.world.observe(actor)
            known={n["id"]:n for n in o["memory"] if n["resource"]==resource}
            known.update({n["id"]:n for n in o["nodes"] if n["resource"]==resource})
            if known:
                return min(known.values(),key=lambda n:(n["x"]-o["self"]["x"])**2+(n["y"]-o["self"]["y"])**2)
            fx,fy=o["senses"]["facing"];a=o["self"]
            self._setup_step(actor,Action("turn",x=max(0,min(24000,a["x"]-fy)),y=max(0,min(24000,a["y"]+fx))))
        raise RuntimeError("Setup could not discover a local resource")

    def _acquire(self,actor,resource,quantity):
        node=self._find(actor,resource);self._approach(actor,node)
        for _ in range(120):
            o=self.world.observe(actor);a=o["self"]
            if a["inventory"][resource]>=quantity:return node
            current=next((n for n in o["nodes"] if n["id"]==node["id"]),None)
            if not current:
                self._setup_step(actor,Action("turn",x=node["x"],y=node["y"]));continue
            amount=min(2,quantity-a["inventory"][resource],current["amount"])
            self._setup_step(actor,Action("gather",target=node["id"],quantity=amount) if amount else Action("rest"))
        raise RuntimeError("Setup resource acquisition timed out")

    def _local_cache(self,actor):
        o=self.world.observe(actor);a=o["self"]
        known={c["id"]:c for c in o["place_memory"] if c["group"]==a["group"]}
        known.update({c["id"]:c for c in o["caches"] if c["group"]==a["group"]})
        if not known:raise RuntimeError("Setup has no legitimately known camp cache")
        return min(known.values(),key=lambda c:(c["x"]-a["x"])**2+(c["y"]-a["y"])**2)

    def _fill_cache(self,actor,target,resource,quantity):
        for _ in range(12):
            self._approach(actor,target)
            cache=next(c for c in self.world.observe(actor)["caches"] if c["id"]==target["id"])
            remaining=quantity-cache["inventory"][resource]
            if remaining<=0:return
            carried=self.world.observe(actor)["self"]["inventory"][resource]
            if carried:
                self._setup_step(actor,Action("deposit",target=target["id"],resource=resource,quantity=min(remaining,carried)))
            else:self._acquire(actor,resource,remaining+2)
        raise RuntimeError("Setup could not deliver the declared cache stock")

    def _build_cache(self,actor):
        self._acquire(actor,2,3);self._acquire(actor,4,2)
        o=self.world.observe(actor);a=o["self"]
        # Search proposed sites by ordinary validation; failed site attempts are
        # themselves recorded actions. No debug geometry/inventory writes.
        for dx,dy in ((250,0),(0,250),(-250,0),(0,-250),(180,180),(-180,180),(180,-180),(-180,-180)):
            point={"x":a["x"]+dx,"y":a["y"]+dy}
            self._setup_step(actor,Action("turn",**point))
            observations=self.world.observations();actions=self.background.actions(observations)
            actions[self.actor]=Action("rest");actions[self.partner]=Action("rest");actions[actor]=Action("place_cache",**point)
            _,results=self._advance(actions,"setup")
            if next(r for r in results if r["actor"]==actor)["ok"]:
                return next(c for c in self.world.observe(actor)["caches"] if c["group"]==a["group"])
        raise RuntimeError("Setup could not build a legitimate cache")

    def _raise_need(self,resource,level):
        other=1-resource
        for _ in range(120):
            o=self.world.observe(self.actor);a=o["self"]
            if a["hunger" if resource==0 else "thirst"]>=level:return
            if a["hunger" if other==0 else "thirst"]>=40:
                if not a["inventory"][other]:self._acquire(self.actor,other,3)
                self._setup_step(self.actor,Action("consume",resource=other))
            else:self._setup_step(self.actor,Action("rest"))
            # The partner's personal food/water must also come from real nodes.
            p=self.world.observe(self.partner)["self"]
            for r,need in ((0,"hunger"),(1,"thirst")):
                if p[need]>=65:
                    if not p["inventory"][r]:self._acquire(self.partner,r,3)
                    self._setup_step(self.partner,Action("consume",resource=r))
        raise RuntimeError("Setup could not establish declared personal need")

    def _setup(self):
        a,p,w=self.actor,self.partner,self.wanted
        if self.task.startswith("invite"):
            self._setup_step(p,Action("depart"));self._setup_step(p,Action("found_group",text="Visitors"))
            target=self._build_cache(p)
            if self.task=="invite_good":
                for r in (0,1):
                    self._fill_cache(p,target,r,4)
            else:
                own=self._local_cache(a)
                for r in (0,1):
                    self._fill_cache(a,own,r,4)
            self._raise_need(w,65)
            self._approach(a,target if self.task=="invite_good" else self._local_cache(a))
            self._approach(p,self.world.observe(a)["self"])
            body=self.world.observe(p)["self"];self._setup_step(a,Action("turn",x=body["x"],y=body["y"]))
            self._setup_step(p,Action("invite",target=a))
        elif self.task.startswith("trade"):
            # In good cases the actor needs the offered resource and owns payment.
            # Bad cases request the actor's last urgent supply for surplus stock.
            desired=w if self.task=="trade_good" else 1-w
            payment=1-desired
            self._acquire(a,payment,4 if self.task=="trade_good" else 2)
            if self.task=="trade_bad":self._acquire(a,desired,4)
            self._acquire(p,desired,6)
            self._raise_need(w,65)
            self._acquire(p,desired,6)
            node=self._find(p,desired);self._approach(p,node)
            # Scarce camp stock is extracted into actual possession, not deleted.
            for _ in range(18):
                o=self.world.observe(p);stock=next(n["amount"] for n in o["nodes"] if n["id"]==node["id"])
                if stock==0 or sum(o["self"]["inventory"])>=22:break
                self._setup_step(p,Action("gather",target=node["id"],quantity=min(2,stock)))
            self._approach(a,self.world.observe(p)["self"])
            body=self.world.observe(p)["self"];self._setup_step(a,Action("turn",x=body["x"],y=body["y"]))
            body=self.world.observe(a)["self"];self._setup_step(p,Action("turn",x=body["x"],y=body["y"]))
            self._setup_step(p,Action("trade",target=a,resource=desired,quantity=2,ask_resource=payment,ask_quantity=1 if self.task=="trade_good" else 2))
        elif self.task=="help_reciprocal":
            self._acquire(a,1-w,4);self._acquire(p,w,6);self._raise_need(w,65)
            self._approach(a,self.world.observe(p)["self"])
            body=self.world.observe(a)["self"];self._setup_step(p,Action("turn",x=body["x"],y=body["y"]))
            self._setup_step(p,Action("request",target=a,resource=1-w,quantity=2))
        for i in self.learner_ids:
            if not self.world.observe(i)["self"]["alive"]:raise RuntimeError("Setup killed a participant")

    def _partner_action(self,o):
        # These are declared partner baselines, not a rule of the game engine.
        a=o["self"];safe=self._safe_action(a["id"])
        if safe:return safe
        if self.partner_kind=="selfish":return self.background.actions({a["id"]:o})[a["id"]]
        if self.task=="help_reciprocal":
            for offer in o["offers"]:
                if offer["to"]==a["id"] and not offer["group"] and not offer["ask_quantity"]:
                    person=next((p for p in o["people"] if p["id"]==offer["from"]),None)
                    if person and (person["x"]-a["x"])**2+(person["y"]-a["y"])**2<=300**2:return Action("accept",target=offer["id"])
            relation=next((r for r in o["relationships"] if r["id"]==self.actor),{})
            recipient=next((p for p in o["people"] if p["id"]==self.actor),None)
            if relation.get("helped",0)>0 and recipient and a["inventory"][self.wanted]>=2:
                if not any(f["from"]==a["id"] and f["to"]==self.actor for f in o["offers"]):
                    if (recipient["x"]-a["x"])**2+(recipient["y"]-a["y"])**2>300**2:return move_toward(o,recipient["x"],recipient["y"])
                    return Action("offer",target=self.actor,resource=self.wanted,quantity=2)
        return self.social.action(o)

    def step(self,actions,partner_policy=None):
        before=self.world.observations();all_actions=self.background.actions(before)
        if self.task=="survival" and partner_policy:
            all_actions.update(partner_policy.actions({i:o for i,o in before.items() if i not in self.learner_ids and o["self"]["alive"]}))
        elif self.task!="survival" and before[self.partner]["self"]["alive"]:
            all_actions[self.partner]=partner_policy.actions({self.partner:before[self.partner]})[self.partner] if partner_policy else self._partner_action(before[self.partner])
        all_actions.update(actions);_,results=self._advance(all_actions,"control");self.decisions+=1
        bad={r["actor"] for r in results if not r["ok"]};self.invalid+=sum(i in bad for i in self.learner_ids)
        after=self.world.observations();rewards={};components={}
        self.useful.extend(self.stock.advance(before,all_actions,results,self.world.debug_state()))
        for result in results:
            if result["ok"] and result["actor"] in self.learner_ids:
                action=all_actions[result["actor"]]
                if action.kind=="accept":self.accepted.append(action.target)
                elif action.kind=="decline":self.declined.append(action.target)
        for i in self.learner_ids:
            old=before[i]["self"];new=after[i]["self"];action=all_actions.get(i,Action())
            if not old["alive"]:
                components[i]={"inactive":0};rewards[i]=0;continue
            cell=(new["x"]//1000,new["y"]//1000);discovery=cell not in self.visited[i];self.visited[i].add(cell)
            c={"survival":0.1 if new["alive"] else -5,"health":(new["health"]-old["health"])*0.1,
               "needs":((old["hunger"]-new["hunger"])+(old["thirst"]-new["thirst"]))*0.012,
               "rest":(old["fatigue"]-new["fatigue"])*0.008,"invalid":-0.1 if i in bad else 0,
               "exploration":0.04 if discovery and max(new["hunger"],new["thirst"])<80 else 0}
            if i not in bad and action.kind=="consume" and action.resource in (0,1):
                need=old["hunger" if action.resource==0 else "thirst"];maximum=45 if action.resource==0 else 55
                c["waste"]=-0.02*max(0,maximum-need)
            components[i]=c;rewards[i]=sum(c.values())
            if new["group"]!=old["group"]:self.memberships.append({"actor":i,"from":old["group"],"to":new["group"],"tick":after[i]["tick"]})
            for r in (0,1):
                delta=new["inventory"][r]-old["inventory"][r]
                if delta>0:self.transfers.append({"actor":i,"resource":r,"quantity":delta,"action":action.kind,"tick":after[i]["tick"]})
        done=self.decisions>=self.horizon or not any(after[i]["self"]["alive"] for i in self.learner_ids)
        self._record({"kind":"reward","step":self.decisions,"components":components})
        return after,rewards,done

    def report(self):
        state=self.world.debug_state();o=self.world.observe(self.actor)
        return {"seed":self.seed,"task":self.task,"partner":self.partner_kind,"population":len(self.world.ids),"steps":self.decisions,
                "setup_ticks":self.start_tick,"alive":sum(p["alive"] for p in state["people"]),
                "learner_alive":all(self.world.observe(i)["self"]["alive"] for i in self.learner_ids),
                "learner_survivors":sum(self.world.observe(i)["self"]["alive"] for i in self.learner_ids),"learner_population":len(self.learner_ids),
                "invalid_actions":self.invalid,"membership_changes":self.memberships,"acquisitions":self.transfers,
                "initial_offers":self.initial_offers,"accepted_offers":self.accepted,"declined_offers":self.declined,
                "useful_consumption":self.useful,
                "final_person":o["self"],"state_hash":self.world.state_hash(),"conserved":state["conserved"]}

    def close(self):
        if self.trace:
            self._record({"kind":"footer",**self.report()});self.trace.close();self.trace=None
        self.world.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def replay_trace(path,policy=None):
    """Verify every setup/control observation, command, result and world hash."""
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        header=json.loads(next(stream))
        if header["kind"]!="header" or header["version"]!=1 or header["native_sha256"]!=native_fingerprint():raise ValueError("Trace runtime/contract mismatch")
        footer=None;steps=0
        with World(snapshot=header["snapshot"]) as world:
            for line in stream:
                if footer is not None:raise ValueError("Records after trace footer")
                record=json.loads(line);kind=record["kind"]
                if kind=="step":
                    if world.observations()!={int(i):o for i,o in record["observations"].items()}:raise ValueError("Trace observation divergence")
                    if policy is not None and record["phase"]=="control":
                        observed=world.observations();expected=policy.actions({i:observed[i] for i in header["learner_ids"] if observed[i]["self"]["alive"]})
                        recorded={int(i):Action(**a) for i,a in record["actions"].items()}
                        if any(recorded.get(i)!=a for i,a in expected.items()):raise ValueError("Frozen neural decision diverged")
                    results=world.step({int(i):Action(**a) for i,a in record["actions"].items()})
                    if results!=record["results"] or world.state_hash()!=record["state_hash"]:raise ValueError("Trace result/hash divergence")
                    steps+=1
                elif kind=="control_start":
                    if world.snapshot()!=record["snapshot"]:raise ValueError("Unrecorded setup state mutation")
                elif kind=="footer":footer=record
                elif kind not in ("reward","decision","controller"):raise ValueError("Unknown trace record")
            if footer is None or footer["state_hash"]!=world.state_hash() or not world.debug_state()["conserved"]:raise ValueError("Incomplete/inconsistent trace")
            return {"steps":steps,"state_hash":world.state_hash(),"conserved":True}
