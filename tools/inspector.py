"""Native desktop diagnostic viewer; this is not the planned first-person client."""
import argparse
import colorsys
import json
import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.native import Action, RESOURCES, ROOT, World
from training.policies import HeuristicPolicy
from training.checkpoint import save_checkpoint,load_checkpoint

BG, PANEL, INK, MUTED = "#101b20", "#19272c", "#e5ece8", "#91a79e"
COLORS = ("#dbaa67", "#6fa2bd", "#aa8a74", "#bab8aa", "#9cac6e")
GROUPS = ("#c9d1cb", "#c8a568", "#7aa6bd", "#a293b9", "#88ad89")

def group_color(group):
    if group<len(GROUPS):return GROUPS[group]
    rgb=colorsys.hls_to_rgb((group*0.61803398875)%1,0.65,0.35)
    return "#"+"".join(f"{round(value*255):02x}" for value in rgb)

class Inspector:
    def __init__(self,root,seed=42,population=32,native_policy=None,social_baseline=False):
        self.root=root;self.seed=seed;self.population=population
        self.world=World(seed,population);self.actor=self.world.ids[0];self.running=False
        self.native_policy=native_policy;self.social_baseline=social_baseline;self.policy=self.make_policy();self.pending=None;self.target=None;self.last_error=""
        root.title("Project Living World — Simulation Inspector")
        root.geometry("1180x800");root.minsize(1000,720);root.configure(bg=BG)
        style=ttk.Style(root);style.theme_use("clam")
        style.configure("TButton",background=PANEL,foreground=INK,padding=(12,8),borderwidth=0)
        style.map("TButton",background=[("active","#30443e")])
        top=tk.Frame(root,bg=BG,padx=24,pady=18);top.pack(fill="x")
        tk.Label(top,text="LIVING WORLD",font=("Segoe UI",20,"bold"),fg=INK,bg=BG).pack(side="left")
        tk.Label(top,text="  /  SIMULATION LAB     v0.3",font=("Segoe UI",10),fg=MUTED,bg=BG).pack(side="left")
        self.status=tk.Label(top,fg="#b6ccac",bg=BG,font=("Consolas",10));self.status.pack(side="right")
        bar=tk.Frame(root,bg=PANEL,padx=18,pady=10);bar.pack(fill="x",padx=24)
        self.run_button=ttk.Button(bar,text="▶ Run",command=self.toggle);self.run_button.pack(side="left",padx=3)
        for label,command in (("Step",self.step),("Save",self.save),("Load",self.load),("New seed",self.reset)):
            ttk.Button(bar,text=label,command=command).pack(side="left",padx=3)
        ttk.Button(bar,text="Social & storage",command=self.social_dialog).pack(side="left",padx=3)
        self.oracle=tk.BooleanVar(value=False);self.manual=tk.BooleanVar(value=False)
        tk.Checkbutton(bar,text="Oracle diagnostics",variable=self.oracle,command=self.draw,bg=PANEL,fg=INK,selectcolor=BG,activebackground=PANEL,activeforeground=INK).pack(side="right",padx=8)
        tk.Checkbutton(bar,text="Control selected individual",variable=self.manual,bg=PANEL,fg=INK,selectcolor=BG,activebackground=PANEL,activeforeground=INK).pack(side="right",padx=8)
        main=tk.Frame(root,bg=BG);main.pack(fill="both",expand=True,padx=24,pady=16)
        left=tk.Frame(main,bg=PANEL,width=278,padx=18,pady=16);left.pack(side="left",fill="y",padx=(0,16));left.pack_propagate(False)
        tk.Label(left,text="INDIVIDUAL",font=("Segoe UI",10,"bold"),fg=MUTED,bg=PANEL).pack(anchor="w")
        self.identity=tk.Label(left,font=("Segoe UI",17,"bold"),fg=INK,bg=PANEL);self.identity.pack(anchor="w",pady=(6,16))
        self.info=tk.Label(left,justify="left",font=("Consolas",11),fg=INK,bg=PANEL);self.info.pack(anchor="w")
        self.inventory=tk.Label(left,justify="left",font=("Consolas",11),fg=INK,bg=PANEL);self.inventory.pack(anchor="w",pady=16)
        tk.Label(left,text="PERSONAL JOURNAL",font=("Segoe UI",10,"bold"),fg=MUTED,bg=PANEL).pack(anchor="w",pady=(10,8))
        self.journal=tk.Label(left,justify="left",anchor="nw",wraplength=240,font=("Segoe UI",10),fg=MUTED,bg=PANEL);self.journal.pack(fill="both",expand=True)
        self.canvas=tk.Canvas(main,bg="#142529",highlightthickness=0);self.canvas.pack(side="left",fill="both",expand=True)
        self.canvas.bind("<Configure>",lambda e:self.draw());self.canvas.bind("<Button-1>",self.click)
        bottom=tk.Frame(root,bg=BG,padx=24,pady=10);bottom.pack(fill="x")
        tk.Label(bottom,text="Click a person to inspect · Manual: click terrain to move · G gather · E eat · D drink · R rest · L leave group",font=("Segoe UI",10),fg=MUTED,bg=BG).pack(anchor="w")
        self.feedback=tk.Label(bottom,text="Only this individual's current sight and remembered sightings are shown.",font=("Segoe UI",9),fg="#b6ccac",bg=BG);self.feedback.pack(anchor="w",pady=5)
        root.bind("<KeyPress>",self.key);root.protocol("WM_DELETE_WINDOW",self.close)
        self.draw();root.after(150,self.loop)

    def close(self):
        self.running=False;self.world.close()
        if hasattr(self.policy,"close"):self.policy.close()
        self.root.destroy()

    def make_policy(self):
        if self.native_policy:
            from inference.runtime import NativeLearnedPolicy
            return NativeLearnedPolicy(self.native_policy)
        if self.social_baseline:
            from training.social import SocialHeuristicPolicy
            return SocialHeuristicPolicy()
        return HeuristicPolicy()

    def toggle(self):
        self.running=not self.running;self.run_button.configure(text="Ⅱ Pause" if self.running else "▶ Run")

    def loop(self):
        if self.running:self.step()
        self.root.after(150,self.loop)

    def step(self):
        observations=self.world.observations();actions=self.policy.actions(observations)
        if self.manual.get():
            actions[self.actor]=self.pending or (Action("move",x=self.target[0],y=self.target[1]) if self.target else Action())
        results=self.world.step(actions);self.pending=None
        failed=next((r for r in results if r["actor"]==self.actor and not r["ok"]),None)
        self.last_error=failed["reason"] if failed else ""
        self.draw()

    def transform(self):
        size=min(self.canvas.winfo_width()-40,self.canvas.winfo_height()-40)
        return max(1,size)/24000,20,20

    def draw(self):
        if not self.world.handle:return
        o=self.world.observe(self.actor);a=o["self"];c=self.canvas;c.delete("all")
        scale,ox,oy=self.transform()
        pos=lambda x,y:(ox+x*scale,oy+y*scale)
        for v in range(0,24001,2000):
            x,y=pos(v,v);c.create_line(x,oy,x,oy+24000*scale,fill="#203235");c.create_line(ox,y,ox+24000*scale,y,fill="#203235")
        if self.oracle.get():
            state=self.world.debug_state();nodes=state["nodes"];people=state["people"];caches=state.get("caches",[])
        else:
            nodes=o["nodes"];people=[a,*o["people"]];caches=o.get("caches",[])
            visible={n["id"] for n in nodes}
            for m in o["memory"]:
                if m["id"] not in visible:
                    x,y=pos(m["x"],m["y"]);c.create_oval(x-3,y-3,x+3,y+3,outline=COLORS[m["resource"]],dash=(2,2))
            visible_caches={cache["id"] for cache in caches}
            for memory in o.get("place_memory",[]):
                if memory["id"] not in visible_caches:
                    x,y=pos(memory["x"],memory["y"]);c.create_rectangle(x-5,y-5,x+5,y+5,outline=group_color(memory["group"]),dash=(2,2))
        x,y=pos(a["x"],a["y"]);radius=1800*scale
        c.create_oval(x-radius,y-radius,x+radius,y+radius,fill="#203a36",outline="#52705e",dash=(3,4))
        for b in o["obstacles"]:
            x,y=pos(b["x"],b["y"]);r=b["radius"]*scale;c.create_oval(x-r,y-r,x+r,y+r,fill="#596260",outline="")
        for n in nodes:
            x,y=pos(n["x"],n["y"]);color=COLORS[n["resource"]]
            c.create_rectangle(x-4,y-4,x+4,y+4,fill=color if n["amount"] else BG,outline=color)
        for cache in caches:
            x,y=pos(cache["x"],cache["y"])
            c.create_rectangle(x-6,y-6,x+6,y+6,fill="#63533e",outline=group_color(cache["group"]),width=2)
            c.create_line(x-4,y,x+4,y,fill="#d1c39a")
        for p in people:
            x,y=pos(p["x"],p["y"]);color=group_color(p["group"]) if p["alive"] else "#5e5e5e"
            c.create_oval(x-4,y-4,x+4,y+4,fill=color,outline=INK if p["id"]==self.actor else "")
            if p["id"]==self.actor:c.create_oval(x-9,y-9,x+9,y+9,outline="#d7dfae",width=2)
        c.create_text(ox+10,oy+12,text="240 m FOREST REGION  /  " + ("ORACLE VIEW" if self.oracle.get() else "LOCAL PERCEPTION"),anchor="nw",fill=MUTED,font=("Segoe UI",10))
        legend="  ".join(f"■ {name}" for name in RESOURCES)
        c.create_text(ox+10,oy+24000*scale-12,text=legend,anchor="sw",fill=MUTED,font=("Segoe UI",9))
        mode="LEARNED SURVIVAL" if self.native_policy else "SOCIAL BASELINE" if self.social_baseline else "BASELINE"
        self.status.configure(text=f"{mode}  ·  SEED {self.seed}  ·  TICK {o['tick']:05}  ·  {'RUNNING' if self.running else 'PAUSED'}")
        self.identity.configure(text=f"Person {self.actor:03}")
        self.info.configure(text=f"Agreement   {a['group'] or 'None'}\nLife        {'Alive' if a['alive'] else 'Deceased'}\nHealth      {a['health']:3}/100\nHunger      {a['hunger']:3}/100\nThirst      {a['thirst']:3}/100\nFatigue     {a['fatigue']:3}/100\nPosition    {a['x']/100:.1f}, {a['y']/100:.1f} m")
        self.inventory.configure(text="INVENTORY  " + str(sum(a["inventory"]))+" / 24\n\n"+"\n".join(f"{name.capitalize():10} {a['inventory'][i]:3}" for i,name in enumerate(RESOURCES)))
        lines=[f"t{e['tick']} · {e['type'].replace('_',' ')}"+(f" ({e['reason']})" if e["reason"] else "") for e in o["events"][:8]]
        self.journal.configure(text="\n\n".join(lines) or "No witnessed events yet.\n\nPeople remember what they actually see.")
        self.feedback.configure(text=self.last_error or ("Oracle diagnostics expose global truth for debugging." if self.oracle.get() else "Only this individual's current sight and remembered sightings are shown."))

    def click(self,event):
        scale,ox,oy=self.transform();x=round((event.x-ox)/scale);y=round((event.y-oy)/scale)
        o=self.world.observe(self.actor)
        people=self.world.debug_state()["people"] if self.oracle.get() else [o["self"],*o["people"]]
        nearest=min(people,key=lambda p:(p["x"]-x)**2+(p["y"]-y)**2)
        if (nearest["x"]-x)**2+(nearest["y"]-y)**2 < (12/scale)**2:
            self.actor=nearest["id"];self.target=None
        elif self.manual.get():
            self.target=(max(0,min(24000,x)),max(0,min(24000,y)))
        self.draw()

    def key(self,event):
        if not self.manual.get():return
        key=event.keysym.lower()
        if key=="g":
            o=self.world.observe(self.actor);a=o["self"]
            n=min(o["nodes"],key=lambda n:(n["x"]-a["x"])**2+(n["y"]-a["y"])**2,default=None)
            if n:self.pending=Action("gather",target=n["id"])
        elif key in ("e","d"):self.pending=Action("consume",resource=0 if key=="e" else 1)
        elif key=="r":self.pending=Action("rest");self.target=None
        elif key=="l":self.pending=Action("depart")
        if not self.running:self.step()

    def save(self):
        if self.native_policy:
            path=filedialog.asksaveasfilename(title="Save a new complete world checkpoint",initialfile="living-world.lwcheck")
            if path:
                try:save_checkpoint(self.world,path,self.policy)
                except (OSError,ValueError) as e:messagebox.showerror("Could not save checkpoint",str(e));return
                self.last_error="Saved world and every individual's decision memory.";self.draw()
            return
        path=filedialog.asksaveasfilename(defaultextension=".lws",initialfile="living-world.lws",filetypes=[("Living World save","*.lws")])
        if path:
            self.world.save(path);self.last_error="Saved world snapshot.";self.draw()

    def load(self):
        path=filedialog.askdirectory(title="Choose a complete world checkpoint") if self.native_policy else filedialog.askopenfilename(filetypes=[("Living World save","*.lws")])
        if not path:return
        try:new=load_checkpoint(path,self.policy,allow_legacy_upgrade=True) if self.native_policy else World(snapshot=Path(path).read_text(encoding="utf-8"))
        except (ValueError,OSError) as e:messagebox.showerror("Could not load save",str(e));return
        self.world.close();self.world=new;self.actor=new.ids[0];self.target=None;self.pending=None
        self.seed=new.debug_state()["seed"];self.last_error="Restored all individuals, resources, offers and memories.";self.draw()

    def reset(self):
        self.world.close();self.seed+=1;self.world=World(self.seed,self.population);self.actor=self.world.ids[0]
        if hasattr(self.policy,"close"):self.policy.close()
        self.policy=self.make_policy()
        self.target=None;self.pending=None;self.last_error="";self.draw()

    def social_dialog(self):
        if self.world.rules_version<2:
            self.world.upgrade_rules();self.last_error="Upgraded this world in memory. Save to a new file to preserve the old version.";self.draw();return
        observation=self.world.observe(self.actor);actor=observation["self"]
        was_running=self.running;self.running=False;self.run_button.configure(text="▶ Run")
        dialog=tk.Toplevel(self.root);dialog.title(f"Person {self.actor} — Social & storage");dialog.configure(bg=PANEL);dialog.resizable(False,False);dialog.transient(self.root);dialog.grab_set()
        frame=tk.Frame(dialog,bg=PANEL,padx=20,pady=16);frame.pack()
        choices=("Request food","Request water","Request visit","Request guard","Answer yes","Answer no","Report result",
                 "Offer gift","Offer trade","Accept offer","Decline offer","Invite to group","Leave group","Found group","Rename group","Deposit","Withdraw","Build cache")
        action=tk.StringVar(value=choices[0]);person=tk.StringVar();resource=tk.StringVar(value="food");quantity=tk.IntVar(value=1)
        payment=tk.StringVar(value="water");payment_quantity=tk.IntVar(value=1);name=tk.StringVar(value="New circle")
        x=tk.StringVar(value=f"{(actor['x']+200)/100:.2f}");y=tk.StringVar(value=f"{actor['y']/100:.2f}")
        people={f"Person {p['id']}":p["id"] for p in observation["people"] if p["alive"]}
        proposal_map={}
        for offer in observation["offers"]:
            if offer["to"]!=actor["id"]:continue
            detail=f"invitation to group {offer['group']}" if offer["group"] else f"{offer['quantity']} {RESOURCES[offer['resource']]}"
            proposal_map[f"Offer {offer['id']} from {offer['from']}: {detail}"]=("offer",offer["id"])
        for request in observation.get("requests",[]):
            if request["to"]==actor["id"] and request["status"]==0:
                proposal_map[f"Request {request['id']} from {request['from']}: {('food','water','visit','guard')[request['intent']]}"]=("request",request["id"])
            elif request["to"]==actor["id"] and request["status"] in (1,3,4):
                proposal_map[f"Task {request['id']}: {('food','water','visit','guard')[request['intent']]}"]=("task",request["id"])
        proposal=tk.StringVar(value=next(iter(proposal_map),""));person.set(next(iter(people),""))
        cache_map={f"Cache {c['id']} — group {c['group']}":c["id"] for c in observation.get("caches",[])}
        cache=tk.StringVar(value=next(iter(cache_map),""))
        def row(number,label,widget):
            tk.Label(frame,text=label,bg=PANEL,fg=INK,font=("Segoe UI",10)).grid(row=number,column=0,sticky="w",padx=(0,16),pady=5)
            widget.grid(row=number,column=1,sticky="ew",pady=5)
        row(0,"Action",ttk.Combobox(frame,textvariable=action,values=choices,state="readonly",width=44))
        row(1,"Visible person",ttk.Combobox(frame,textvariable=person,values=list(people),state="readonly"))
        row(2,"Incoming proposal",ttk.Combobox(frame,textvariable=proposal,values=list(proposal_map),state="readonly"))
        row(3,"Visible cache",ttk.Combobox(frame,textvariable=cache,values=list(cache_map),state="readonly"))
        row(4,"Resource",ttk.Combobox(frame,textvariable=resource,values=RESOURCES,state="readonly"))
        row(5,"Quantity",tk.Spinbox(frame,from_=1,to=24,textvariable=quantity))
        row(6,"Trade payment",ttk.Combobox(frame,textvariable=payment,values=RESOURCES,state="readonly"))
        row(7,"Payment quantity",tk.Spinbox(frame,from_=1,to=24,textvariable=payment_quantity))
        row(8,"Group name",tk.Entry(frame,textvariable=name))
        destination=tk.Frame(frame,bg=PANEL);tk.Entry(destination,textvariable=x,width=14).pack(side="left");tk.Entry(destination,textvariable=y,width=14).pack(side="left",padx=8)
        row(9,"Destination (metres)",destination)
        tk.Label(frame,text="This pauses the lab. Requests can be refused. A cache needs 3 wood, 2 fiber and a clear site within reach.",bg=PANEL,fg=MUTED,wraplength=490,justify="left").grid(row=10,column=0,columnspan=2,sticky="w",pady=12)
        closed=False
        def finish():
            nonlocal closed
            if closed:return
            closed=True;dialog.destroy();self.running=was_running;self.run_button.configure(text="Ⅱ Pause" if was_running else "▶ Run");self.draw()
        def submit():
            try:
                selected=action.get();r=RESOURCES.index(resource.get());q=quantity.get()
                target=people.get(person.get(),0)
                if selected.startswith("Request "):
                    intent=("Request food","Request water","Request visit","Request guard").index(selected)
                    result=Action("request",target=target,resource=intent,quantity=q if intent<2 else 1,x=round(float(x.get())*100),y=round(float(y.get())*100))
                elif selected.startswith("Answer "):
                    kind,identifier=proposal_map.get(proposal.get(),("",0))
                    if kind!="request":raise ValueError("Select an incoming request")
                    result=Action("answer_request",target=identifier,quantity=int(selected=="Answer yes"))
                elif selected=="Report result":
                    kind,identifier=proposal_map.get(proposal.get(),("",0))
                    if kind!="task":raise ValueError("Select an accepted task")
                    result=Action("report_request",target=identifier)
                elif selected in ("Accept offer","Decline offer"):
                    kind,identifier=proposal_map.get(proposal.get(),("",0))
                    if kind!="offer":raise ValueError("Select an incoming offer")
                    result=Action("accept" if selected=="Accept offer" else "decline",target=identifier)
                elif selected in ("Offer gift","Offer trade"):
                    result=Action("offer" if selected=="Offer gift" else "trade",target=target,resource=r,quantity=q,ask_resource=RESOURCES.index(payment.get()),ask_quantity=payment_quantity.get())
                elif selected=="Invite to group":result=Action("invite",target=target)
                elif selected=="Leave group":result=Action("depart")
                elif selected in ("Found group","Rename group"):result=Action("found_group" if selected=="Found group" else "rename_group",text=name.get())
                elif selected in ("Deposit","Withdraw"):
                    if cache.get() not in cache_map:raise ValueError("Select a visible cache")
                    result=Action(selected.lower(),target=cache_map[cache.get()],resource=r,quantity=q)
                else:result=Action("place_cache",x=round(float(x.get())*100),y=round(float(y.get())*100))
            except (ValueError,OverflowError,tk.TclError) as error:messagebox.showerror("Check action",str(error),parent=dialog);return
            self.manual.set(True);self.pending=result;self.target=None;finish();self.step()
        buttons=tk.Frame(frame,bg=PANEL);buttons.grid(row=11,column=0,columnspan=2,sticky="e")
        ttk.Button(buttons,text="Cancel",command=finish).pack(side="left",padx=5)
        ttk.Button(buttons,text="Act as this person",command=submit).pack(side="left")
        dialog.protocol("WM_DELETE_WINDOW",finish)

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=42);p.add_argument("--population",type=int,default=32)
    p.add_argument("--native-policy");p.add_argument("--social-baseline",action="store_true");args=p.parse_args()
    if args.native_policy and args.social_baseline:raise SystemExit("Choose one controller")
    root=tk.Tk();Inspector(root,args.seed,args.population,args.native_policy,args.social_baseline);root.mainloop()

if __name__=="__main__":main()
