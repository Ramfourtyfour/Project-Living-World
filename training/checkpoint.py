"""Immutable world/controller checkpoint bundles with an atomic publication marker.

A directory without manifest.json is incomplete and is never loaded. Publishing
the final manifest makes the already-fsynced world and neural state one checkpoint.
Existing checkpoints are never overwritten. Hashes detect corruption, not an
attacker able to rewrite the entire trusted local checkpoint.
"""
import hashlib
import json
import os
from pathlib import Path

from .native import World, native_fingerprint

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")

def controller_record(policy):
    if policy is None:
        return {"id":"external-v1","memory":None}
    identifier=getattr(policy,"name",None)
    if not isinstance(identifier,str) or not identifier:
        raise ValueError("Controller must declare a stable name/version")
    memory=policy.memory_snapshot() if hasattr(policy,"memory_snapshot") else None
    return {"id":identifier,"memory":memory}

def restore_controller(policy,record,ids):
    if record["memory"] is not None:
        if policy is None or not hasattr(policy,"restore_memory"):
            raise ValueError("This checkpoint requires its matching neural policy")
        if controller_record(policy)["id"]!=record["id"]:
            raise ValueError("Controller version differs from checkpoint")
        memory=record["memory"]
        if not isinstance(memory,dict) or not isinstance(memory.get("hidden"),dict):
            raise ValueError("Malformed controller state")
        parsed=[int(i) for i in memory["hidden"]]
        if len(set(parsed))!=len(parsed) or any(str(i)!=key for i,key in zip(parsed,memory["hidden"])):
            raise ValueError("Noncanonical recurrent identity")
        if set(parsed)-set(ids):
            raise ValueError("Recurrent memory references unknown individuals")
        policy.restore_memory(memory)
    elif policy is not None and controller_record(policy)["id"]!=record["id"]:
        raise ValueError("Controller version differs from checkpoint")

def _write_exclusive(path,data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def save_checkpoint(world,path,policy=None):
    path=Path(path)
    record=controller_record(policy)
    if record["memory"] is not None and set(map(int,record["memory"]["hidden"]))-set(world.ids):
        raise ValueError("Controller memory references another world")
    files={"world.lws":world.snapshot().encode("utf-8"),"controller.json":canonical(record)}
    metadata={"version":1,"native_sha256":native_fingerprint(),"state_hash":world.state_hash(),
              "individual_ids":list(world.ids),"files":{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    if getattr(world,"migration_info",None):metadata["migration"]=world.migration_info
    # mkdir is an exclusive claim, preventing two writers from replacing a bundle.
    path.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        _write_exclusive(path/name,data)
    _write_exclusive(path/"manifest.pending",canonical(metadata))
    os.replace(path/"manifest.pending",path/"manifest.json")
    return metadata

def _load_checkpoint(path,policy=None,allow_legacy_upgrade=False):
    path=Path(path)
    try:
        manifest=path/"manifest.json"
        if manifest.is_symlink() or manifest.stat().st_size>65536:
            raise ValueError("Invalid checkpoint manifest")
        raw=manifest.read_bytes()
    except FileNotFoundError as error:
        raise ValueError("Checkpoint is incomplete: no published manifest") from error
    if len(raw)>65536:
        raise ValueError("Checkpoint manifest too large")
    metadata=json.loads(raw)
    if metadata["version"]!=1 or set(metadata["files"])!={"world.lws","controller.json"}:
        raise ValueError("Unsupported checkpoint contract")
    runtime_changed=metadata["native_sha256"]!=native_fingerprint()
    if runtime_changed and not allow_legacy_upgrade:
        raise ValueError("Native runtime differs; explicit save migration is required")
    files={}
    for name,expected in metadata["files"].items():
        source=path/name
        if source.is_symlink() or source.stat().st_size>16*1024*1024:
            raise ValueError("Invalid checkpoint file")
        files[name]=source.read_bytes()
        if hashlib.sha256(files[name]).hexdigest()!=expected:
            raise ValueError(f"Checkpoint integrity mismatch: {name}")
    record=json.loads(files["controller.json"])
    world=World(snapshot=files["world.lws"].decode("utf-8"))
    try:
        if list(world.ids)!=metadata["individual_ids"] or world.state_hash()!=metadata["state_hash"]:
            raise ValueError("Checkpoint world identity/hash mismatch")
        if runtime_changed and world.rules_version!=1:
            raise ValueError("Native runtime differs; this rules version has no verified migration")
        restore_controller(policy,record,world.ids)
        if "migration" in metadata:
            if not isinstance(metadata["migration"],dict):raise ValueError("Malformed migration provenance")
            world.migration_info=metadata["migration"]
        if runtime_changed:
            world.upgrade_rules()
            world.migration_info={"from_rules":1,"to_rules":2,"previous_native_sha256":metadata["native_sha256"],
                                  "previous_state_hash":metadata["state_hash"],"source_manifest_sha256":hashlib.sha256(raw).hexdigest()}
    except BaseException:
        world.close()
        raise
    return world

def load_checkpoint(path,policy=None,allow_legacy_upgrade=False):
    try:
        return _load_checkpoint(path,policy,allow_legacy_upgrade)
    except (KeyError,TypeError,AttributeError,UnicodeError,RecursionError,OverflowError) as error:
        raise ValueError("Malformed checkpoint contract") from error
