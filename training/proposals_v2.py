"""Version 2 adds factual own-cost inputs; it does not rank/mask choices by utility."""
from dataclasses import replace
from .proposals import BASE_SIZE,MAX_CANDIDATES,KINDS,Proposal,context,teacher_index
from .proposals import proposals as legacy_proposals

CANDIDATE_SIZE=64


def proposals(o,relationships=True):
    old=legacy_proposals(o,relationships);a=o["self"];inventory=a["inventory"]
    if not a["alive"]:return [replace(p,features=p.features+(0.0,)*12) for p in old]
    stock=[m for m in o["place_memory"] if m["group"]==a["group"] and m["inventory_at_observation"] is not None]
    reserves=[sum(m["inventory_at_observation"][r] for m in stock) for r in (0,1)]
    needs=(a["hunger"]/100,a["thirst"]/100);result=[]
    for p in old:
        f=p.features;giving=next((r for r in range(5) if f[25+r]),None);receiving=next((r for r in range(5) if f[30+r]),None)
        projected=list(inventory)
        if p.kind not in ("decline","answer_no"):
            if giving is not None:projected[giving]-=round(f[35]*4)
            if receiving is not None:projected[receiving]+=round(f[36]*4)
        membership=p.kind=="leave" or (p.kind=="accept" and f[48] and not f[40])
        extra=(needs[giving] if giving in (0,1) else 0,needs[receiving] if receiving in (0,1) else 0,
               inventory[giving]/8 if giving is not None else 0,inventory[receiving]/8 if receiving is not None else 0,
               max(-3,min(3,projected[0]/8)),max(-3,min(3,projected[1]/8)),*needs,
               min(2,reserves[0]/12),min(2,reserves[1]/12),min(2,sum(reserves)/12) if membership else 0,float(membership))
        result.append(replace(p,features=f+extra))
    return result
