"""Native desktop diagnostic viewer; this is not the planned first-person client."""
import argparse
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

class Inspector:
    def __init__(self,root,seed=42,population=32,native_policy=None):
        self.root=root;self.seed=seed;self.population=population
        self.world=World(seed,population);self.actor=self.world.ids[0];self.running=False
        self.native_policy=native_policy;self.policy=self.make_policy();self.pending=None;self.target=None;self.last_error=""
        root.title("Project Living World — Simulation Inspector")
        root.geometry("1180x800");root.minsize(1000,720);root.configure(bg=BG)
        style=ttk.Style(root);style.theme_use("clam")
        style.configure("TButton",background=PANEL,foreground=INK,padding=(12,8),borderwidth=0)
        style.map("TButton",background=[("active","#30443e")])
        top=tk.Frame(root,bg=BG,padx=24,pady=18);top.pack(fill="x")
        tk.Label(top,text="LIVING WORLD",font=("Segoe UI",20,"bold"),fg=INK,bg=BG).pack(side="left")
        tk.Label(top,text="  /  SIMULATION LAB     v0.2",font=("Segoe UI",10),fg=MUTED,bg=BG).pack(side="left")
        self.status=tk.Label(top,fg="#b6ccac",bg=BG,font=("Consolas",10));self.status.pack(side="right")
        bar=tk.Frame(root,bg=PANEL,padx=18,pady=10);bar.pack(fill="x",padx=24)
        self.run_button=ttk.Button(bar,text="▶ Run",command=self.toggle);self.run_button.pack(side="left",padx=3)
        for label,command in (("Step",self.step),("Save",self.save),("Load",self.load),("New seed",self.reset)):
            ttk.Button(bar,text=label,command=command).pack(side="left",padx=3)
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
            state=self.world.debug_state();nodes=state["nodes"];people=state["people"]
        else:
            nodes=o["nodes"];people=[a,*o["people"]]
            visible={n["id"] for n in nodes}
            for m in o["memory"]:
                if m["id"] not in visible:
                    x,y=pos(m["x"],m["y"]);c.create_oval(x-3,y-3,x+3,y+3,outline=COLORS[m["resource"]],dash=(2,2))
        x,y=pos(a["x"],a["y"]);radius=1800*scale
        c.create_oval(x-radius,y-radius,x+radius,y+radius,fill="#203a36",outline="#52705e",dash=(3,4))
        for b in o["obstacles"]:
            x,y=pos(b["x"],b["y"]);r=b["radius"]*scale;c.create_oval(x-r,y-r,x+r,y+r,fill="#596260",outline="")
        for n in nodes:
            x,y=pos(n["x"],n["y"]);color=COLORS[n["resource"]]
            c.create_rectangle(x-4,y-4,x+4,y+4,fill=color if n["amount"] else BG,outline=color)
        for p in people:
            x,y=pos(p["x"],p["y"]);color=GROUPS[p["group"]] if p["alive"] else "#5e5e5e"
            c.create_oval(x-4,y-4,x+4,y+4,fill=color,outline=INK if p["id"]==self.actor else "")
            if p["id"]==self.actor:c.create_oval(x-9,y-9,x+9,y+9,outline="#d7dfae",width=2)
        c.create_text(ox+10,oy+12,text="240 m FOREST REGION  /  " + ("ORACLE VIEW" if self.oracle.get() else "LOCAL PERCEPTION"),anchor="nw",fill=MUTED,font=("Segoe UI",10))
        legend="  ".join(f"■ {name}" for name in RESOURCES)
        c.create_text(ox+10,oy+24000*scale-12,text=legend,anchor="sw",fill=MUTED,font=("Segoe UI",9))
        self.status.configure(text=f"{'LEARNED' if self.native_policy else 'BASELINE'}  ·  SEED {self.seed}  ·  TICK {o['tick']:05}  ·  {'RUNNING' if self.running else 'PAUSED'}")
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
        try:new=load_checkpoint(path,self.policy) if self.native_policy else World(snapshot=Path(path).read_text(encoding="utf-8"))
        except (ValueError,OSError) as e:messagebox.showerror("Could not load save",str(e));return
        self.world.close();self.world=new;self.actor=new.ids[0];self.target=None;self.pending=None
        self.seed=new.debug_state()["seed"];self.last_error="Restored all individuals, resources, offers and memories.";self.draw()

    def reset(self):
        self.world.close();self.seed+=1;self.world=World(self.seed,self.population);self.actor=self.world.ids[0]
        if hasattr(self.policy,"close"):self.policy.close()
        self.policy=self.make_policy()
        self.target=None;self.pending=None;self.last_error="";self.draw()

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=42);p.add_argument("--population",type=int,default=32)
    p.add_argument("--native-policy");args=p.parse_args()
    root=tk.Tk();Inspector(root,args.seed,args.population,args.native_policy);root.mainloop()

if __name__=="__main__":main()
