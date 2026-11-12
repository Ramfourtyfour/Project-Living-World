"""Standard shared recurrent context + shared candidate scorer, not a faction policy."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from .proposals_v2 import BASE_SIZE,CANDIDATE_SIZE,MAX_CANDIDATES,KINDS,context,proposals

ARCHITECTURE="social-gru64-slate-v2"


class SocialModel(nn.Module):
    hidden_size=64

    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(BASE_SIZE,64),nn.Tanh())
        self.memory=nn.GRU(64,64)
        self.survival_actor=nn.Linear(64,8)
        self.scorer=nn.Sequential(nn.Linear(64+CANDIDATE_SIZE,64),nn.Tanh(),nn.Linear(64,1))
        self.critic=nn.Linear(64,1)

    def forward(self,x,candidates,hidden=None):
        state,hidden=self.memory(self.encoder(x),hidden)
        repeated=state.unsqueeze(-2).expand(*candidates.shape[:-1],64)
        logits=self.scorer(torch.cat((repeated,candidates),dim=-1)).squeeze(-1)
        logits=logits+(self.survival_actor(state).unsqueeze(-2)*candidates[...,:8]).sum(-1)
        return logits,self.critic(state).squeeze(-1),hidden

    def initialize_survival(self,path):
        from .learn import LearnedPolicy
        old=LearnedPolicy(path)
        if old.model.architecture!="gru64-local-v2":raise ValueError("Transfer requires the frozen GRU64 local survival model")
        with torch.no_grad():
            self.encoder[0].weight.zero_();self.encoder[0].weight[:,:38].copy_(old.model.encoder[0].weight)
            self.encoder[0].bias.copy_(old.model.encoder[0].bias)
            self.memory.load_state_dict(old.model.memory.state_dict())
            self.survival_actor.load_state_dict(old.model.actor.state_dict())
            self.scorer[-1].weight.zero_();self.scorer[-1].bias.zero_()
            self.critic.weight.zero_();self.critic.bias.zero_()
        return old.manifest["sha256"]

    def initialize_social(self,path):
        from .social_model import LearnedSocialPolicy as PriorPolicy
        old=PriorPolicy(path);previous=old.model.state_dict();current=self.state_dict()
        with torch.no_grad():
            for name,value in current.items():
                if name=="scorer.0.weight":value.zero_();value[:,:116].copy_(previous[name])
                else:value.copy_(previous[name])
        self.load_state_dict(current)
        return old.manifest["sha256"]


def tensors(observations,ids,relationships=True):
    slates=[proposals(observations[i],relationships) for i in ids]
    x=torch.tensor([context(observations[i],relationships) for i in ids],dtype=torch.float32)
    candidates=torch.zeros(len(ids),MAX_CANDIDATES,CANDIDATE_SIZE)
    masks=torch.zeros(len(ids),MAX_CANDIDATES,dtype=torch.bool)
    for j,slate in enumerate(slates):
        candidates[j,:len(slate)]=torch.tensor([p.features for p in slate])
        masks[j,:len(slate)]=torch.tensor([p.enabled for p in slate])
    return x,candidates,masks,slates


def save_artifact(model,path,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix(".json").exists():raise FileExistsError("Preserve prior social artifacts")
    torch.save(model.state_dict(),path)
    manifest={"version":1,"architecture":ARCHITECTURE,"observation_version":3,"proposal_version":2,
              "base_inputs":BASE_SIZE,"candidate_inputs":CANDIDATE_SIZE,"hidden":64,"max_candidates":MAX_CANDIDATES,
              "candidate_kinds":list(KINDS),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),**metadata}
    path.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


class LearnedSocialPolicy:
    def __init__(self,path,*,relationships=True,memory=True):
        path=Path(path)
        if path.stat().st_size>4*1024*1024 or path.with_suffix(".json").stat().st_size>1024*1024:raise ValueError("Oversized social artifact")
        self.manifest=json.loads(path.with_suffix(".json").read_bytes());m=self.manifest
        if (m["version"]!=1 or m["architecture"]!=ARCHITECTURE or m["observation_version"]!=3 or m["proposal_version"]!=2
            or m["base_inputs"]!=BASE_SIZE or m["candidate_inputs"]!=CANDIDATE_SIZE or m["hidden"]!=64 or m["max_candidates"]!=MAX_CANDIDATES
            or m["candidate_kinds"]!=list(KINDS)):raise ValueError("Unsupported social policy contract")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=m["sha256"]:raise ValueError("Social policy checksum mismatch")
        self.model=SocialModel();self.model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));self.model.eval()
        self.hidden={};self.relationships=relationships;self.use_memory=memory
        self.name=f"{ARCHITECTURE}-{m['sha256']}-relations{int(relationships)}-memory{int(memory)}"

    @torch.no_grad()
    def decisions(self,observations):
        ids=[i for i,o in observations.items() if o["self"]["alive"]]
        if not ids:return {},{}
        x,candidates,mask,slates=tensors(observations,ids,self.relationships)
        hidden=torch.stack([self.hidden.get(i,torch.zeros(64)) if self.use_memory else torch.zeros(64) for i in ids]).unsqueeze(0)
        logits,_,new=self.model(x.unsqueeze(0),candidates.unsqueeze(0),hidden)
        chosen=logits[0].masked_fill(~mask,-1e9).argmax(-1).tolist()
        self.hidden.update({i:new[0,j].clone() for j,i in enumerate(ids)})
        return {i:slates[j][chosen[j]].action for j,i in enumerate(ids)},{i:slates[j][chosen[j]].kind for j,i in enumerate(ids)}

    def actions(self,observations):return self.decisions(observations)[0]

    def memory_snapshot(self):
        return {"policy_sha256":self.manifest["sha256"],"relationships":self.relationships,"use_memory":self.use_memory,
                "hidden":{str(i):h.tolist() for i,h in sorted(self.hidden.items())}}

    def restore_memory(self,snapshot):
        if snapshot["policy_sha256"]!=self.manifest["sha256"] or snapshot["relationships"]!=self.relationships or snapshot["use_memory"]!=self.use_memory:
            raise ValueError("Social controller version differs from memory")
        updated={}
        for raw,row in snapshot["hidden"].items():
            if not isinstance(raw,str) or not raw.isascii() or not raw.isdecimal() or str(int(raw))!=raw or not 0<int(raw)<2**64:raise ValueError("Invalid social memory identity")
            if not isinstance(row,list) or len(row)!=64:raise ValueError("Invalid social memory shape")
            h=torch.tensor(row,dtype=torch.float32)
            if not torch.isfinite(h).all() or h.abs().max()>1.00001:raise ValueError("Invalid social memory value")
            updated[int(raw)]=h
        if len(updated)>4096:raise ValueError("Social memory capacity exceeded")
        self.hidden=updated
