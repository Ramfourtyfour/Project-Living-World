"""Versioned survival transfer, local demonstrations and recurrent PPO for C2."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import torch
from torch.distributions import Categorical

from .native import native_fingerprint
from .proposals import KINDS,teacher_index
from .proposal_baselines import ProposalHeuristicPolicy
from .social_arena import Arena,TASKS
from .social_model import SocialModel,LearnedSocialPolicy,tensors,save_artifact
from .policies import HeuristicPolicy

ROOT=Path(__file__).resolve().parents[1]


def collect_demo(seed,task,horizon,path):
    teacher=ProposalHeuristicPolicy();xs=[];cs=[];ms=[];ys=[];ks=[];valid=[];matches=0;total=0
    with Arena(seed,task,horizon=horizon,trace=path) as arena:
        ids=arena.learner_ids
        for _ in range(horizon):
            o=arena.world.observations();x,c,mask,slates=tensors(o,ids)
            chosen=[teacher_index(o[i],slates[j],teacher) if o[i]["self"]["alive"] else 7 for j,i in enumerate(ids)]
            live=[o[i]["self"]["alive"] for i in ids]
            matches+=sum(live[j] and slates[j][chosen[j]].action==teacher.action(o[i]) for j,i in enumerate(ids));total+=sum(live)
            actions={i:slates[j][chosen[j]].action for j,i in enumerate(ids) if live[j]}
            arena._record({"kind":"decision","controller":teacher.name,"choices":{str(i):slates[j][chosen[j]].kind for j,i in enumerate(ids) if live[j]}})
            _,_,done=arena.step(actions)
            xs.append(x);cs.append(c);ms.append(mask);ys.append(torch.tensor(chosen));valid.append(torch.tensor(live,dtype=torch.float32))
            ks.append(torch.tensor([KINDS.index(slates[j][chosen[j]].kind) for j in range(len(ids))]))
            if done:break
        report=arena.report()
    return {"x":torch.stack(xs),"c":torch.stack(cs),"mask":torch.stack(ms),"target":torch.stack(ys),
            "kinds":torch.stack(ks),"valid":torch.stack(valid),"report":report,"teacher_exact_matches":matches,"teacher_decisions":total}


def clone(model,data,epochs):
    # Protect transferred survival dynamics during initial supervised learning;
    # new observation columns and the candidate scorer can learn social context.
    for p in model.memory.parameters():p.requires_grad_(False)
    for p in model.survival_actor.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam([{"params":model.scorer.parameters(),"lr":0.001},
                                {"params":model.encoder.parameters(),"lr":0.0002}])
    counts=torch.zeros(len(KINDS))
    for d in data:counts+=torch.bincount(d["kinds"][d["valid"].bool()],minlength=len(KINDS))
    weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights=weights/weights[counts>0].mean()
    records=[]
    for epoch in range(epochs):
        random.shuffle(data);correct=0;total=0;losses=[]
        for d in data:
            hidden=None
            for start in range(0,len(d["x"]),32):
                sl=slice(start,start+32);valid=d["valid"][sl]
                logits,_,hidden=model(d["x"][sl],d["c"][sl],hidden)
                logits=logits.masked_fill(~d["mask"][sl],-1e9)
                raw=torch.nn.functional.cross_entropy(logits.flatten(0,1),d["target"][sl].flatten(),reduction="none").reshape(valid.shape)
                weight=weights[d["kinds"][sl]]*valid
                loss=(raw*weight).sum()/weight.sum().clamp(min=1e-6)
                optimizer.zero_grad();loss.backward()
                if model.encoder[0].weight.grad is not None:model.encoder[0].weight.grad[:,:38].zero_()
                if model.encoder[0].bias.grad is not None:model.encoder[0].bias.grad.zero_()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();hidden=hidden.detach()
                correct+=int(((logits.argmax(-1)==d["target"][sl])*valid).sum());total+=int(valid.sum());losses.append(float(loss.detach()))
        row={"bc_epoch":epoch+1,"accuracy":correct/max(1,total),"loss":sum(losses)/len(losses)};records.append(row)
        if epoch==0 or (epoch+1)%4==0:print(json.dumps(row),flush=True)
    for p in model.parameters():p.requires_grad_(True)
    return {"class_counts":dict(zip(KINDS,counts.tolist())),"history":records}


def rollout(model,arena,partner_policy=None):
    ids=arena.learner_ids;hidden=torch.zeros(1,len(ids),64)
    xs=[];cs=[];masks=[];choices=[];logs=[];values=[];rewards=[];continues=[];valid=[]
    for _ in range(arena.horizon):
        o=arena.world.observations();x,c,mask,slates=tensors(o,ids)
        with torch.no_grad():
            logits,value,new=model(x[None],c[None],hidden)
            dist=Categorical(logits=logits[0].masked_fill(~mask,-1e9));chosen=dist.sample();log=dist.log_prob(chosen)
        live=torch.tensor([o[i]["self"]["alive"] for i in ids],dtype=torch.float32)
        actions={i:slates[j][int(chosen[j])].action for j,i in enumerate(ids) if live[j]}
        arena._record({"kind":"decision","controller":"current-recurrent-ppo","choices":{str(i):slates[j][int(chosen[j])].kind for j,i in enumerate(ids) if live[j]}})
        after,r,done=arena.step(actions,partner_policy)
        active=torch.tensor([after[i]["self"]["alive"] for i in ids],dtype=torch.float32)
        xs.append(x);cs.append(c);masks.append(mask);choices.append(chosen);logs.append(log);values.append(value[0])
        rewards.append(torch.tensor([r[i] for i in ids]));continues.append(active);valid.append(live)
        hidden=new*active[None,:,None]
        if done:break
    return {"x":torch.stack(xs),"c":torch.stack(cs),"mask":torch.stack(masks),"choices":torch.stack(choices),
            "logp":torch.stack(logs),"value":torch.stack(values),"reward":torch.stack(rewards),
            "continues":torch.stack(continues),"valid":torch.stack(valid),"report":arena.report()}


def ppo_update(model,optimizer,d,epochs):
    reward=d["reward"];value=d["value"];active=d["continues"];valid=d["valid"]
    advantage=torch.zeros_like(reward);carry=torch.zeros(reward.shape[1]);next_value=torch.zeros_like(carry)
    for t in reversed(range(len(reward))):
        delta=reward[t]+0.99*next_value*active[t]-value[t]
        carry=delta+0.99*0.95*active[t]*carry;advantage[t]=carry;next_value=value[t]
    returns=advantage+value;selected=advantage[valid.bool()]
    normalized=(advantage-selected.mean())/(selected.std(unbiased=False)+1e-6)
    last_loss=0
    for _ in range(epochs):
        hidden=None;logits=[];estimates=[]
        for t in range(len(reward)):
            score,v,hidden=model(d["x"][t:t+1],d["c"][t:t+1],hidden)
            hidden=hidden*active[t][None,:,None];logits.append(score[0]);estimates.append(v[0])
        dist=Categorical(logits=torch.stack(logits).masked_fill(~d["mask"],-1e9))
        ratio=(dist.log_prob(d["choices"])-d["logp"]).exp()
        actor=-torch.minimum(ratio*normalized,ratio.clamp(0.8,1.2)*normalized)
        critic=(torch.stack(estimates)-returns).square()
        loss=((actor+0.05*critic-0.003*dist.entropy())*valid).sum()/valid.sum()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),0.5);optimizer.step();last_loss=float(loss.detach())
    return last_loss


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=31)
    p.add_argument("--bc-episodes",type=int,default=24);p.add_argument("--bc-epochs",type=int,default=24)
    p.add_argument("--updates",type=int,default=96);p.add_argument("--horizon",type=int,default=64)
    p.add_argument("--ppo-epochs",type=int,default=4);p.add_argument("--init",default="inference/artifacts/curriculum-v3.pt")
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();source={str(f.relative_to(ROOT)):f.read_bytes() for f in [*Path(__file__).parent.glob("*.py"),*ROOT.glob("sim/*.cpp"),*ROOT.glob("sim/*.hpp"),*ROOT.glob("sim/*.h")]}
    with zipfile.ZipFile(output/"source.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name,data in source.items():archive.writestr(name,data)
    configuration={**vars(args),"native_sha256":native_fingerprint(),"training_partition":"below 45000","validation_partition":"50000..50015","sealed_partition":"60000..60031",
                   "source_sha256":{k:hashlib.sha256(v).hexdigest() for k,v in source.items()},"source_bundle_sha256":hashlib.sha256((output/"source.zip").read_bytes()).hexdigest()}
    (output/"configuration.json").write_text(json.dumps(configuration,indent=2)+"\n",encoding="utf-8")
    model=SocialModel();configuration["transfer_policy_sha256"]=model.initialize_survival(args.init)
    (output/"configuration.json").write_text(json.dumps(configuration,indent=2)+"\n",encoding="utf-8")
    save_artifact(model,output/"initial.pt",{"phase":"survival transfer; untrained social head","configuration":configuration})
    data=[]
    for episode in range(args.bc_episodes):
        task=TASKS[episode%len(TASKS)];horizon=512 if task=="survival" else args.horizon
        seed=1000+args.seed+5*episode+episode//len(TASKS)
        d=collect_demo(seed,task,horizon,output/"demonstrations"/f"{episode:04}-{task}.jsonl.gz");data.append(d)
        print(json.dumps({"demonstration":episode+1,"task":task,"decisions":d["teacher_decisions"],"exact_teacher_matches":d["teacher_exact_matches"]}),flush=True)
    bc=clone(model,data,args.bc_epochs);del data
    save_artifact(model,output/"bc.pt",{"phase":"local demonstration warm start","configuration":configuration,"bc":bc})
    optimizer=torch.optim.Adam([{"params":[*model.encoder.parameters(),*model.memory.parameters(),*model.survival_actor.parameters()],"lr":0.000003},
                                {"params":model.scorer.parameters(),"lr":0.0003},{"params":model.critic.parameters(),"lr":0.001}])
    history=[];league=[output/"bc.pt"]
    for update in range(args.updates):
        task=TASKS[update%len(TASKS)];seed=2000+args.seed+5*update+update//len(TASKS)
        mode=(update//len(TASKS)+update%len(TASKS))%3
        partner_kind="selfish" if mode==1 else "reciprocal"
        partner=LearnedSocialPolicy(random.choice(league)) if mode==2 else HeuristicPolicy() if mode==1 and task=="survival" else None
        with Arena(seed,task,horizon=256 if task=="survival" else args.horizon,trace=output/"rollouts"/f"{update:04}-{task}.jsonl.gz",partner=partner.name if partner else partner_kind,
                   learner_fraction=0.5 if task=="survival" and partner else 1) as arena:
            arena._record({"kind":"controller","training_update":update+1,"partner":partner.name if partner else partner_kind})
            d=rollout(model,arena,partner)
        loss=ppo_update(model,optimizer,d,args.ppo_epochs)
        row={"update":update+1,"task":task,"reward_per_person":float(d["reward"].sum()/d["reward"].shape[1]),
             "alive_fraction":float(d["continues"][-1].mean()),"loss":loss,**d["report"]}
        history.append(row);(output/"history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
        if update==0 or (update+1)%4==0:print(json.dumps({k:row[k] for k in ("update","task","reward_per_person","alive_fraction","invalid_actions","membership_changes")}),flush=True)
        if (update+1)%16==0:
            snapshot=output/f"snapshot-{update+1:04}.pt";save_artifact(model,snapshot,{"phase":"recurrent PPO snapshot","configuration":configuration,"updates":update+1});league.append(snapshot)
    save_artifact(model,output/"policy.pt",{"phase":"recurrent PPO candidate; unevaluated","configuration":configuration,"bc":bc,"updates":args.updates,"seconds":time.perf_counter()-started,
                 "history_sha256":hashlib.sha256((output/"history.json").read_bytes()).hexdigest()})
    print(json.dumps({"candidate":str(output/"policy.pt"),"seconds":time.perf_counter()-started}),flush=True)


if __name__=="__main__":main()
