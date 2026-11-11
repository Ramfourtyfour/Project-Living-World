"""Explicit, original-preserving rules migration after exact old-world replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from training.checkpoint import save_checkpoint
from training.journal import JournalRunner
from training.native import native_fingerprint
from training.policies import HeuristicPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("checkpoint")
    p.add_argument("--native-policy");p.add_argument("--new-journal");a=p.parse_args()
    policy=HeuristicPolicy()
    if a.native_policy:
        from inference.runtime import NativeLearnedPolicy
        policy=NativeLearnedPolicy(a.native_policy)
    source=Path(a.source);source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    with JournalRunner.open(source,policy,allow_runtime_migration=True) as verified:
        old_hash=verified.world.state_hash();old_rules=verified.world.rules_version
        before=verified.world.debug_state();identities=verified.world.ids
        verified.world.upgrade_rules()
        after=verified.world.debug_state()
        if before["ledger"]!=after["ledger"] or identities!=verified.world.ids:
            raise RuntimeError("Migration changed material accounting or identities")
        verified.world.migration_info={"from_rules":old_rules,"to_rules":verified.world.rules_version,
                                      "previous_state_hash":old_hash,"source_journal_sha256":source_sha}
        metadata=save_checkpoint(verified.world,a.checkpoint,policy)
        report={"version":1,"source_sha256":source_sha,"source_preserved":True,"previous_rules":old_rules,
                "rules":verified.world.rules_version,"previous_state_hash":old_hash,"state_hash":verified.world.state_hash(),
                "native_sha256":native_fingerprint(),"individual_ids":list(identities),"ledger_preserved":True,
                "cache_count":len(after.get("caches",[]))}
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:raise RuntimeError("Original journal changed")
    path=Path(a.checkpoint)/"migration.json";path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if a.new_journal:
        with JournalRunner.from_checkpoint(a.new_journal,a.checkpoint,policy):pass
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
