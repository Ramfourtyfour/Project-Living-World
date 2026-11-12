"""Versioned, decentralized parameterized action interface for social learning.

IDs resolve observed handles only. Features never contain numeric identities or
another person's private state. Physical feasibility is distinct from usefulness.
"""
from dataclasses import dataclass
import math
from .native import Action
from .policies import execute_intent,intent_mask,features_v2,move_toward,facing_target

BASE_SIZE=64
CANDIDATE_SIZE=52
MAX_CANDIDATES=128
KINDS=("food","water","eat","drink","rest","explore","wood","wait","look","approach",
       "offer","trade","request","accept","decline","answer_yes","answer_no","invite",
       "leave","found","deposit","withdraw","report","call","crouch")


def distance2(a,b):return (a["x"]-b["x"])**2+(a["y"]-b["y"])**2


def context(o,relationships=True):
    if o["version"]!=3:raise ValueError("Social proposal encoder requires rules-3 observations")
    a=o["self"]
    if not a["alive"]:return [0.0]*BASE_SIZE
    senses=o["senses"];offers=o["offers"];requests=o["requests"]
    relations=o["relationships"] if relationships else []
    trusts=[r.get("trust",0)/100 for r in relations]
    caches=[c for c in o["caches"] if c["group"]==a["group"] and "inventory" in c]
    old=[m for m in o["place_memory"] if m["group"]==a["group"] and m["inventory_at_observation"] is not None]
    heard=o["heard"]
    f=features_v2(o)+[senses["facing"][0]/1000,senses["facing"][1]/1000,senses["ambient_light"]/100,
                     senses["cover_density"]/100,float(senses["crouching"])]
    f += [min(1,sum(c["kind"]==kind for c in heard)/8) for kind in ("footsteps","work","voice","call")]
    f += [min(1,sum(r["to"]==a["id"] for r in offers)/8),min(1,sum(r["from"]==a["id"] for r in offers)/8),
          min(1,sum(r["to"]==a["id"] and r["status"]==0 for r in requests)/8),
          min(1,sum(r["to"]==a["id"] and r["status"]==1 for r in requests)/8)]
    f += [sum(trusts)/len(trusts) if trusts else 0,min(trusts,default=0),max(trusts,default=0),
          min(1,sum(r.get("helped",0) for r in relations)/16),min(1,sum(r.get("broken",0) for r in relations)/16)]
    f += [float(a["group"]!=0),min(1,len(caches)/4)]
    f += [min(2,sum(c["inventory"][r] for c in caches)/12) for r in (0,1)]
    f += [min(2,sum(c["inventory_at_observation"][r] for c in old)/12) for r in (0,1)]
    f += [min(1,len(o["person_memory"])/16),min((c["age"]/40 for c in heard),default=1)]
    assert len(f)==BASE_SIZE
    return f


@dataclass(frozen=True)
class Proposal:
    kind:str
    action:Action
    features:tuple
    enabled:bool=True


def proposals(o,relationships=True):
    if o["version"]!=3:raise ValueError("Social proposals require rules 3")
    a=o["self"];inventory=a["inventory"];result=[]
    people={p["id"]:p for p in o["people"] if p["alive"]}
    memories={p["id"]:p for p in o["person_memory"]}
    relations={r["id"]:r for r in o["relationships"]} if relationships else {}

    def add(kind,action,*,partner=None,point=None,give=None,receive=None,group=0,request=None,offer=None,enabled=True):
        if len(result)>=MAX_CANDIDATES:return
        v=[0.0]*CANDIDATE_SIZE;v[KINDS.index(kind)]=1.0
        if give is not None:
            r,q=give
            if 0<=r<5:v[25+r]=1.0
            v[35]=min(6,q/4)
        if receive is not None:
            r,q=receive
            if 0<=r<5:v[30+r]=1.0
            v[36]=min(6,q/4)
        position=point or partner
        if position is not None:
            v[37]=max(-2,min(2,(position["x"]-a["x"])/1800))
            v[38]=max(-2,min(2,(position["y"]-a["y"])/1800))
            v[39]=min(2,math.sqrt(distance2(a,position))/1800)
        other_group=group or (partner.get("group",partner.get("group_at_observation",0)) if partner else 0)
        v[40]=float(bool(a["group"]) and other_group==a["group"])
        v[41]=float(bool(other_group))
        v[42]=float(bool(other_group) and any(c["group"]==other_group for c in [*o["caches"],*o["place_memory"]]))
        if position is not None:v[45]=min(1,position.get("age",0)/128)
        if partner is not None:
            relation=relations.get(partner["id"],{})
            v[43]=relation.get("trust",0)/100;v[44]=relation.get("familiarity",0)/100
            v[45]=min(1,partner.get("age",0)/128)
        if request:
            v[46]=max(0,min(1,(request["quantity"]-request["progress"])/max(1,request["quantity"])))
            v[47]=max(0,min(1,(request["expires"]-o["tick"])/60));v[51]=request["intent"]/3
        if offer:
            v[47]=max(0,min(1,(offer["expires"]-o["tick"])/40))
            v[48]=float(bool(offer["group"]));v[49]=float(not offer["group"] and not offer["ask_quantity"])
            v[50]=float(not offer["group"] and bool(offer["ask_quantity"]))
        result.append(Proposal(kind,action,tuple(v),bool(enabled)))

    mask=intent_mask(o) if a["alive"] else [False]*7+[True]
    for i,kind in enumerate(KINDS[:8]):
        add(kind,execute_intent(o,i) if a["alive"] else Action(),enabled=mask[i])
    if not a["alive"]:return result
    # Unconditional individual choices have no trust/usefulness thresholds.
    add("leave" if a["group"] else "found",Action("depart" if a["group"] else "found_group"))
    add("call",Action("call"));add("crouch",Action("crouch",quantity=int(not o["senses"]["crouching"])))
    for dx,dy in ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)):
        point={"x":max(0,min(24000,a["x"]+dx*1000)),"y":max(0,min(24000,a["y"]+dy*1000))}
        if distance2(a,point):add("look",Action("turn",x=point["x"],y=point["y"]),point=point)

    for offer in sorted(o["offers"],key=lambda f:f["id"]):
        if offer["to"]!=a["id"]:continue
        sender=people.get(offer["from"]);known=sender or memories.get(offer["from"])
        give=(offer["ask_resource"],offer["ask_quantity"]) if offer["ask_quantity"] else None
        receive=None if offer["group"] else (offer["resource"],offer["quantity"])
        data={"partner":known,"give":give,"receive":receive,"group":offer["group"],"offer":offer}
        add("decline",Action("decline",target=offer["id"]),**data)
        legal=sender and distance2(a,sender)<=300**2 and offer["expires"]>o["tick"]
        if offer["group"]:legal=legal and sender["group"]==offer["group"]
        else:
            legal=legal and (not give or inventory[give[0]]>=give[1]) and sum(inventory)+offer["quantity"]-offer["ask_quantity"]<=24
        if legal:add("accept",Action("accept",target=offer["id"]),**data)
        if known and not sender:
            add("look",Action("turn",x=known["x"],y=known["y"]),point=known,partner=known)
        elif sender and distance2(a,sender)>300**2:
            add("approach",move_toward(o,sender["x"],sender["y"]),partner=sender)

    for request in sorted(o["requests"],key=lambda r:r["id"]):
        if request["to"]!=a["id"]:continue
        sender=people.get(request["from"]);known=sender or memories.get(request["from"])
        data={"partner":known,"request":request,"give":(request["intent"],request["quantity"]) if request["intent"]<2 else None}
        reachable=sender and distance2(a,sender)<=1200**2
        if request["status"]==0 and request["expires"]>o["tick"] and reachable:
            add("answer_yes",Action("answer_request",target=request["id"],quantity=1),**data)
            add("answer_no",Action("answer_request",target=request["id"],quantity=0),**data)
        if request["status"] in (1,3,4) and reachable:
            add("report",Action("report_request",target=request["id"]),**data)
        if request["status"] in (0,1,3):
            point={"x":request["contact_x"],"y":request["contact_y"]}
            if request["status"]==1 and request["intent"]>=2:point=request
            if distance2(a,point)>100**2:
                action=move_toward(o,point["x"],point["y"]) if facing_target(o,point["x"],point["y"]) else Action("turn",x=point["x"],y=point["y"])
                add("approach" if action.kind=="move" else "look",action,point=point,**data)

    visible_cache_ids={c["id"] for c in o["caches"] if "inventory" in c}
    for memory in sorted(o["place_memory"],key=lambda c:(distance2(a,c),c["id"]))[:4]:
        if memory["group"]!=a["group"] or memory["id"] in visible_cache_ids:continue
        if not facing_target(o,memory["x"],memory["y"]):action=Action("turn",x=memory["x"],y=memory["y"])
        else:action=move_toward(o,memory["x"],memory["y"])
        add("look" if action.kind=="turn" else "approach",action,point=memory,group=memory["group"])

    for cache in sorted(o["caches"],key=lambda c:(distance2(a,c),c["id"]))[:4]:
        if cache["group"]!=a["group"] or "inventory" not in cache:continue
        for resource in range(5):
            for q in (1,2):
                if cache["inventory"][resource]>=q and sum(inventory)+q<=24:
                    add("withdraw",Action("withdraw",target=cache["id"],resource=resource,quantity=q),point=cache,receive=(resource,q),group=cache["group"])
                if inventory[resource]>=q and sum(cache["inventory"])+q<=120:
                    add("deposit",Action("deposit",target=cache["id"],resource=resource,quantity=q),point=cache,give=(resource,q),group=cache["group"])

    outgoing=sum(f["from"]==a["id"] for f in o["offers"])
    pending=sum(r["from"]==a["id"] and r["status"] in (0,1) for r in o["requests"])
    partners=sorted(people.values(),key=lambda p:(distance2(a,p),p["id"]))[:4]
    for partner in partners:
        distance=distance2(a,partner)
        add("approach",move_toward(o,partner["x"],partner["y"]),partner=partner)
        if distance<=300**2 and outgoing<8:
            if a["group"]:add("invite",Action("invite",target=partner["id"]),partner=partner,group=a["group"])
            for resource in (0,1):
                other=1-resource
                for q in (1,2):
                    if inventory[resource]>=q:
                        add("offer",Action("offer",target=partner["id"],resource=resource,quantity=q),partner=partner,give=(resource,q))
                        for payment in (1,2):
                            add("trade",Action("trade",target=partner["id"],resource=resource,quantity=q,ask_resource=other,ask_quantity=payment),partner=partner,give=(resource,q),receive=(other,payment))
        if distance<=1200**2 and pending<8:
            for resource in (0,1):
                for q in (1,2):add("request",Action("request",target=partner["id"],resource=resource,quantity=q),partner=partner,receive=(resource,q))
    return result


def teacher_index(observation,slate,teacher):
    action=teacher.action(observation)
    for i,p in enumerate(slate):
        if p.enabled and p.action==action:return i
    # A finite teacher need not cover every low-level waypoint. Match its declared
    # survival intent rather than inventing an optimal social label.
    intent=teacher.survival.intent(observation)
    return intent if slate[intent].enabled else 7
