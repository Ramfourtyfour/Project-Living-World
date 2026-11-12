"""Evaluation-only FIFO lineage for fungible stock, never a policy observation.

Native conservation remains authoritative. This trace allocates actual acquired
units to actual transfers/consumption so repeated offers cannot count as aid.
"""
class StockTrace:
    def __init__(self,state):
        self.serial=0;self.bags={};self.caches={};self.consumed=[];self.sunk=[]
        for p in state["people"]:self.bags[p["id"]]=[[self._new("setup",p["id"]) for _ in range(q)] for q in p["inventory"]]
        for c in state["caches"]:self.caches[c["id"]]=[[self._new("setup_cache",c["id"]) for _ in range(q)] for q in c["inventory"]]

    def _new(self,via,source):
        self.serial+=1
        return (self.serial,via,source)

    @staticmethod
    def _take(container,resource,quantity):
        if len(container[resource])<quantity:raise RuntimeError("Stock lineage underflow")
        taken=container[resource][:quantity];del container[resource][:quantity]
        return taken

    @staticmethod
    def _put(container,resource,units,via,source):
        container[resource].extend((root,via,source) for root,_,_ in units)

    def advance(self,before,actions,results,after_state):
        used=[]
        for result in results:
            if not result["ok"]:continue
            actor=result["actor"];q=actions[actor];bag=self.bags[actor];o=before[actor]
            if q.kind=="gather":
                node=next(n for n in o["nodes"] if n["id"]==q.target)
                bag[node["resource"]].extend(self._new("gather",actor) for _ in range(q.quantity))
            elif q.kind=="consume":
                unit=self._take(bag,q.resource,1)[0];self.consumed.append(unit[0])
                need=o["self"]["hunger" if q.resource==0 else "thirst"]
                used.append({"actor":actor,"resource":q.resource,"root":unit[0],"via":unit[1],"source":unit[2],
                             "useful_need_reduction":min(45 if q.resource==0 else 55,need)})
            elif q.kind=="accept":
                offer=next(f for f in o["offers"] if f["id"]==q.target)
                if offer["group"]:continue
                giver=offer["from"];other=self.bags[giver]
                units=self._take(other,offer["resource"],offer["quantity"])
                payment=self._take(bag,offer["ask_resource"],offer["ask_quantity"]) if offer["ask_quantity"] else []
                kind="trade" if payment else "gift"
                self._put(bag,offer["resource"],units,kind,giver)
                if payment:self._put(other,offer["ask_resource"],payment,"trade",actor)
            elif q.kind in ("deposit","withdraw"):
                cache=self.caches.setdefault(q.target,[[] for _ in range(5)])
                source,target=(bag,cache) if q.kind=="deposit" else (cache,bag)
                self._put(target,q.resource,self._take(source,q.resource,q.quantity),"cache",q.target)
            elif q.kind=="place_cache":
                for resource,quantity in ((2,3),(4,2)):self.sunk.extend(u[0] for u in self._take(bag,resource,quantity))
        for cache in after_state["caches"]:self.caches.setdefault(cache["id"],[[] for _ in range(5)])
        for person in after_state["people"]:
            if [len(v) for v in self.bags[person["id"]]]!=person["inventory"]:raise RuntimeError("Personal stock lineage diverged")
        for cache in after_state["caches"]:
            if [len(v) for v in self.caches[cache["id"]]]!=cache["inventory"]:raise RuntimeError("Cache stock lineage diverged")
        roots=[u[0] for containers in (self.bags,self.caches) for bag in containers.values() for stack in bag for u in stack]
        roots+=self.consumed+self.sunk
        if len(roots)!=len(set(roots)) or len(roots)!=self.serial:raise RuntimeError("Duplicated/lost stock lineage")
        return used
