"""Verify a legacy complete checkpoint and publish a separate rules-2 bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.checkpoint import controller_record,load_checkpoint,save_checkpoint
from training.journal import JournalRunner
from training.native import native_fingerprint
from training.policies import HeuristicPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("destination")
    p.add_argument("--native-policy");p.add_argument("--new-journal");p.add_argument("--report")
    args=p.parse_args();source=Path(args.source)
    # Only known bundle files; source bytes are never rewritten.
    original={name:(source/name).read_bytes() for name in ("manifest.json","world.lws","controller.json")}
    metadata=json.loads(original["manifest.json"])
    policy=HeuristicPolicy()
    if args.native_policy:
        from inference.runtime import NativeLearnedPolicy
        policy=NativeLearnedPolicy(args.native_policy)
    with load_checkpoint(source,policy,allow_legacy_upgrade=True) as world:
        if world.rules_version==1:world.upgrade_rules()
        if not getattr(world,"migration_info",None):
            world.migration_info={"from_rules":1 if original["world.lws"].startswith(b"LIVING_WORLD_V1") else 2,
                                  "to_rules":world.rules_version,"previous_native_sha256":metadata["native_sha256"],
                                  "previous_state_hash":metadata["state_hash"],
                                  "source_manifest_sha256":hashlib.sha256(original["manifest.json"]).hexdigest()}
        preserved=controller_record(policy)==json.loads(original["controller.json"])
        if not preserved:raise RuntimeError("Controller memory changed during migration")
        save_checkpoint(world,args.destination,policy)
        report={"version":1,"source_state_hash":metadata["state_hash"],"source_native_sha256":metadata["native_sha256"],
                "native_sha256":native_fingerprint(),"rules_version":world.rules_version,"state_hash":world.state_hash(),
                "cache_count":len(world.debug_state().get("caches",[])),"controller_memory_preserved":preserved,
                "original_preserved":all((source/name).read_bytes()==data for name,data in original.items())}
        if not report["original_preserved"]:raise RuntimeError("Source checkpoint changed during migration")
    if args.new_journal:
        with JournalRunner.from_checkpoint(args.new_journal,args.destination,policy):pass
    if args.report:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if hasattr(policy,"close"):policy.close()
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
