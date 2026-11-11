"""Single-writer durable authority with replayable, hash-linked action commits.

Each proposed tick runs on a snapshot copy. Its complete record is fsynced before
the authoritative world pointer changes. A write failure poisons that runner;
reopen/replay establishes what reached disk. An incomplete/corrupt record is never
silently dropped. The log is private server data, not a client observation stream.
"""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

from .checkpoint import canonical,controller_record,restore_controller,load_checkpoint
from .native import Action,World,native_fingerprint

ZERO="0"*64
MAX_RECORD=32*1024*1024

class _Lease:
    def __init__(self,path,create):
        self.stream=Path(path).open("x+b" if create else "r+b")
        try:
            if sys.platform=="win32":
                import msvcrt
                self.stream.seek(0);msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        self.stream.close() # Kernel releases the file lease, including on process death.

class JournalRunner:
    def __init__(self):
        self.world=None;self.policy=None;self._lease=None;self._previous=ZERO;self._broken=False
        self._controller_id=None
        self._read_only=False;self.controller_snapshot=None

    @classmethod
    def create(cls,path,seed=42,population=32,scenario=0,policy=None,rules_version=2):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        runner=cls();runner.policy=policy
        try:
            controller=controller_record(policy)
            if controller["memory"] is not None and controller["memory"]["hidden"]:
                raise ValueError("A new world requires fresh controller memory; reopen its journal to resume")
            runner._lease=_Lease(path,True)
            runner.world=World(seed,population,scenario=scenario,rules_version=rules_version)
            runner._controller_id=controller["id"]
            runner._append({"kind":"header","version":1,"native_sha256":native_fingerprint(),
                            "snapshot":runner.world.snapshot(),"controller":controller})
        except BaseException as error:
            runner.close()
            if isinstance(error,(KeyError,TypeError,AttributeError,UnicodeError,RecursionError,OverflowError)):
                raise ValueError("Malformed journal contract") from error
            raise
        return runner

    @classmethod
    def open(cls,path,policy=None,trace=None,allow_runtime_migration=False,restore_policy=True):
        runner=cls();runner.policy=policy
        try:
            runner._lease=_Lease(path,False)
            stream=runner._lease.stream;stream.seek(0)
            last_controller=None;records=0
            while True:
                line=stream.readline(MAX_RECORD+1)
                if not line:break
                if len(line)>MAX_RECORD or not line.endswith(b"\n"):
                    raise ValueError("Journal has an oversized or incomplete record; explicit recovery required")
                envelope=json.loads(line)
                payload=envelope["payload"]
                digest=hashlib.sha256(runner._previous.encode()+canonical(payload)).hexdigest()
                if envelope["previous"]!=runner._previous or envelope["sha256"]!=digest:
                    raise ValueError("Journal chain/integrity mismatch")
                if records==0:
                    if payload["kind"]!="header" or payload["version"]!=1:
                        raise ValueError("Unsupported journal contract")
                    if payload["native_sha256"]!=native_fingerprint():
                        if not allow_runtime_migration:raise ValueError("Journal runtime differs; explicit migration required")
                        runner._read_only=True
                    runner.world=World(snapshot=payload["snapshot"])
                    if payload.get("migration") is not None:
                        if not isinstance(payload["migration"],dict):raise ValueError("Malformed migration provenance")
                        runner.world.migration_info=payload["migration"]
                    runner._controller_id=payload["controller"]["id"]
                    if trace is not None:trace(deepcopy(payload),runner.world.observations())
                else:
                    if payload["kind"]!="step" or payload["controller"]["id"]!=runner._controller_id:
                        raise ValueError("Invalid journal step/controller transition")
                    actions={int(a["actor"]):Action(**a["action"]) for a in payload["actions"]}
                    if len(actions)!=len(payload["actions"]):
                        raise ValueError("Duplicate journal actor")
                    requests={int(a["actor"]):int(a["request"]) for a in payload["actions"]}
                    if trace is not None:trace(deepcopy(payload),runner.world.observations())
                    results=runner.world.step(actions,requests=requests)
                    if results!=payload["results"] or runner.world.state_hash()!=payload["state_hash"]:
                        raise ValueError("Journal replay diverged")
                last_controller=payload["controller"]
                runner._previous=digest;records+=1
            if not records:
                raise ValueError("Journal is incomplete: no header")
            runner.controller_snapshot=last_controller
            if restore_policy:restore_controller(policy,last_controller,runner.world.ids)
            else:runner._read_only=True
            stream.seek(0,os.SEEK_END)
        except BaseException as error:
            runner.close()
            if isinstance(error,(KeyError,TypeError,AttributeError,UnicodeError,RecursionError,OverflowError)):
                raise ValueError("Malformed journal contract") from error
            raise
        return runner

    @classmethod
    def from_checkpoint(cls,path,checkpoint,policy=None):
        """Begin a new journal from a fully verified world/controller bundle."""
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        runner=cls();runner.policy=policy
        try:
            runner.world=load_checkpoint(checkpoint,policy)
            runner._lease=_Lease(path,True)
            controller=controller_record(policy);runner._controller_id=controller["id"]
            runner._append({"kind":"header","version":1,"native_sha256":native_fingerprint(),
                            "snapshot":runner.world.snapshot(),"controller":controller,
                            "migration":getattr(runner.world,"migration_info",None),
                            "checkpoint_sha256":hashlib.sha256((Path(checkpoint)/"manifest.json").read_bytes()).hexdigest()})
        except BaseException:
            runner.close();raise
        return runner

    def _append(self,payload):
        digest=hashlib.sha256(self._previous.encode()+canonical(payload)).hexdigest()
        line=canonical({"previous":self._previous,"sha256":digest,"payload":payload})+b"\n"
        if len(line)>MAX_RECORD:
            raise ValueError("Journal record too large")
        try:
            stream=self._lease.stream
            if stream.write(line)!=len(line):raise OSError("Short journal write")
            stream.flush();os.fsync(stream.fileno())
        except BaseException:
            self._broken=True
            raise
        self._previous=digest

    def advance(self,actions):
        if self._read_only:raise RuntimeError("Migration/audit replay is read-only; start a new checkpoint/journal")
        if self._broken or self.world is None or self._lease is None:
            raise RuntimeError("Runner unavailable; close and reopen/replay the journal")
        if set(actions)-set(self.world.ids):
            raise ValueError("Action for unknown individual")
        controller=controller_record(self.policy)
        if controller["id"]!=self._controller_id:
            raise ValueError("Controller version changed; start a new journal/checkpoint")
        requests={i:self.world.observe(i)["self"]["last_request"]+1 for i in actions}
        proposed=World(snapshot=self.world.snapshot())
        if getattr(self.world,"migration_info",None):proposed.migration_info=deepcopy(self.world.migration_info)
        try:
            results=proposed.step(actions,requests=requests)
            self._append({"kind":"step","actions":[{"actor":i,"request":requests[i],"action":asdict(actions[i])} for i in sorted(actions)],
                          "results":results,"state_hash":proposed.state_hash(),"controller":controller})
        except BaseException:
            proposed.close()
            raise
        previous=self.world;self.world=proposed;previous.close()
        return results

    def advance_policy(self,overrides=None):
        if self._read_only:raise RuntimeError("Migration/audit replay is read-only; start a new checkpoint/journal")
        if self._broken or self.world is None or self._lease is None:
            raise RuntimeError("Runner unavailable; close and reopen/replay the journal")
        if self.policy is None:
            raise ValueError("No controller attached")
        previous=controller_record(self.policy)
        try:
            observations=self.world.observations()
            actions=self.policy.actions(observations)
            actions.update(overrides or {})
            return self.advance(actions)
        except BaseException:
            restore_controller(self.policy,previous,self.world.ids)
            raise

    def close(self):
        if self.world is not None:self.world.close();self.world=None
        if self._lease is not None:self._lease.close();self._lease=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def recover_incomplete_tail(source,target,policy=None):
    """Explicitly recover an interrupted final write into a new verified log.

    Never alter the original, never skip a complete corrupt record, and never
    overwrite a target. The caller can inspect the original tail and recovery
    report. The single-writer lease prevents racing an active authority.
    """
    source,target=Path(source),Path(target)
    if source.resolve()==target.resolve():
        raise ValueError("Recovery must preserve the original journal")
    lease=_Lease(source,False)
    try:
        stream=lease.stream;stream.seek(0);previous=ZERO;prefix_bytes=0;total_bytes=0;whole=hashlib.sha256();tail=None
        while True:
            line=stream.readline(MAX_RECORD+1)
            if not line:break
            whole.update(line);total_bytes+=len(line)
            if len(line)>MAX_RECORD:
                raise ValueError("Oversized record cannot be repaired by tail recovery")
            if not line.endswith(b"\n"):
                tail=line;break
            envelope=json.loads(line);payload=envelope["payload"]
            digest=hashlib.sha256(previous.encode()+canonical(payload)).hexdigest()
            if envelope["previous"]!=previous or envelope["sha256"]!=digest:
                raise ValueError("Complete corrupt record cannot be discarded by tail recovery")
            previous=digest;prefix_bytes+=len(line)
        if tail is None or prefix_bytes==0:
            raise ValueError("No recoverable incomplete final record")
        target.parent.mkdir(parents=True,exist_ok=True)
        stream.seek(0)
        with target.open("xb") as destination:
            remaining=prefix_bytes
            while remaining:
                chunk=stream.read(min(1024*1024,remaining))
                if not chunk:raise OSError("Journal changed during leased recovery")
                destination.write(chunk);remaining-=len(chunk)
            destination.flush();os.fsync(destination.fileno())
        with JournalRunner.open(target,policy) as verified:
            metadata={"version":1,"source_sha256":whole.hexdigest(),"original_bytes":total_bytes,
                      "recovered_bytes":prefix_bytes,"incomplete_tail_bytes":len(tail),
                      "last_state_hash":verified.world.state_hash(),
                      "last_tick":verified.world.observe(verified.world.ids[0])["tick"],
                      "original_preserved":True}
        report=target.with_name(target.name+".recovery.json")
        with report.open("xb") as destination:
            destination.write(canonical(metadata));destination.flush();os.fsync(destination.fileno())
        return metadata
    finally:
        lease.close()
