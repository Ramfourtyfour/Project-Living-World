"""Declared individual social heuristic baseline; it is not a learned social policy.

Every choice uses only that individual's current observation and received proposals.
No registry inventory queries, faction commands or scripted membership splits.
"""
from .native import Action
from .policies import HeuristicPolicy,execute_intent,move_toward

def distance2(a,b):return (a["x"]-b["x"])**2+(a["y"]-b["y"])**2

class SocialHeuristicPolicy:
    name="individual-social-heuristic-v1"

    def __init__(self):self.survival=HeuristicPolicy()

    def action(self,o):
        a=o["self"];inventory=a["inventory"];people={p["id"]:p for p in o["people"] if p["alive"]}
        if a["thirst"]>=75 and inventory[1]:return Action("consume",resource=1)
        if a["hunger"]>=75 and inventory[0]:return Action("consume",resource=0)
        if a["fatigue"]>=85:return Action("rest")
        # Offers describe only volunteered quantities, never another private bag.
        for offer in o["offers"]:
            if offer["to"]!=a["id"]:continue
            sender=people.get(offer["from"])
            if not sender or distance2(a,sender)>300**2:continue
            failed=any(e["actor"]==a["id"] and e["type"]=="rejected" and e["target"]==offer["id"] and e["reason"] in ("insufficient_items","inventory_full","expired_offer") for e in o["events"])
            if failed:return Action("decline",target=offer["id"])
            if offer["group"]:
                infrastructure=any(c["group"]==offer["group"] for c in o.get("caches",[]))
                return Action("accept" if not a["group"] and infrastructure else "decline",target=offer["id"])
            resource=offer["resource"];ask=offer["ask_resource"];payment=offer["ask_quantity"]
            room=sum(inventory)+offer["quantity"]-payment<=24
            useful=resource in (0,1) and inventory[resource]<3
            affordable=not payment or (inventory[ask]>=payment+(1 if ask in (0,1) else 0))
            return Action("accept" if room and useful and affordable else "decline",target=offer["id"])
        for request in o.get("requests",[]):
            if request["to"]!=a["id"] or request["status"]!=0:continue
            sender=people.get(request["from"])
            if not sender or distance2(a,sender)>1200**2:continue
            intent=request["intent"]
            can_help=(intent<2 and inventory[intent]>=request["quantity"]+1) or (intent>=2 and max(a["hunger"],a["thirst"])<60)
            return Action("answer_request",target=request["id"],quantity=int(can_help))
        for request in o.get("requests",[]):
            if request["to"]==a["id"] and request["status"]==3 and request.get("report_tick",0)<request.get("completed_tick",0):
                partner=people.get(request["from"])
                if partner and distance2(a,partner)<=1200**2:return Action("report_request",target=request["id"])
                contact={"x":request["contact_x"],"y":request["contact_y"]}
                if distance2(a,contact)>300**2:return move_toward(o,contact["x"],contact["y"])
        for request in o.get("requests",[]):
            if request["to"]!=a["id"] or request["status"]!=1:continue
            intent=request["intent"];recipient=people.get(request["from"])
            remaining=request["quantity"]-request["progress"]
            if intent<2 and inventory[intent]>=remaining+1 and recipient:
                duplicate=any(f["from"]==a["id"] and f["to"]==recipient["id"] and not f["group"] and f["resource"]==intent for f in o["offers"])
                if duplicate:continue
                if distance2(a,recipient)>300**2:return move_toward(o,recipient["x"],recipient["y"])
                return Action("offer",target=recipient["id"],resource=intent,quantity=remaining)
            if intent<2 and inventory[intent]>=remaining+1 and not recipient:
                if distance2(a,request)>300**2:return move_toward(o,request["x"],request["y"])
            if intent>=2 and max(a["hunger"],a["thirst"],a["fatigue"])<75:
                if distance2(a,request)<=300**2:return Action("rest")
                return move_toward(o,request["x"],request["y"])
        # Current reachable contents are authorized observations. Remembered stock
        # may be stale; it never becomes a remote withdrawal operation.
        for cache in o.get("caches",[]):
            if cache["group"]!=a["group"] or "inventory" not in cache:continue
            for resource in (0,1):
                if inventory[resource]<2 and cache["inventory"][resource]>0:
                    return Action("withdraw",target=cache["id"],resource=resource,quantity=min(2-inventory[resource],cache["inventory"][resource]))
                if inventory[resource]>4:
                    return Action("deposit",target=cache["id"],resource=resource,quantity=inventory[resource]-3)
        # Ask a visible person when immediate personal supply is failing. The
        # recipient can refuse and continues using its own controller.
        for resource,need in ((1,a["thirst"]),(0,a["hunger"])):
            pending=any(r["from"]==a["id"] and r["intent"]==resource and r["status"] in (0,1) for r in o.get("requests",[]))
            if need>=70 and inventory[resource]==0 and not pending:
                nearby=[p for p in people.values() if distance2(a,p)<=1200**2]
                if nearby:
                    other=min(nearby,key=lambda p:(distance2(a,p),p["id"]))
                    return Action("request",target=other["id"],resource=resource)
        return execute_intent(o,self.survival.intent(o))

    def actions(self,observations):
        return {i:self.action(o) for i,o in observations.items() if o["self"]["alive"]}
