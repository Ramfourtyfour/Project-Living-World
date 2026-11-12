import importlib.util
import json
import subprocess
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.journal import JournalRunner,recover_incomplete_tail
from training.native import Action
from training.policies import HeuristicPolicy
from training.checkpoint import save_checkpoint

def test_journal_replays_and_appends_without_duplicate_items(tmp_path):
    path=tmp_path/"world.lwj"
    with JournalRunner.create(path,42,16,policy=HeuristicPolicy()) as runner:
        for _ in range(32):runner.advance_policy()
        saved=runner.world.snapshot()
        with pytest.raises(OSError):JournalRunner.open(path)
    with JournalRunner.open(path,HeuristicPolicy()) as runner:
        assert runner.world.snapshot()==saved
        for _ in range(16):runner.advance_policy()
        final=runner.world.state_hash()
        assert runner.world.debug_state()["conserved"]
    with JournalRunner.open(path,HeuristicPolicy()) as runner:
        assert runner.world.state_hash()==final

def test_rejected_request_is_durable_and_bad_batch_does_not_mutate(tmp_path):
    path=tmp_path/"world.lwj"
    with JournalRunner.create(path,42,4) as runner:
        before=runner.world.state_hash();size=path.stat().st_size
        with pytest.raises(ValueError):runner.advance({99999:Action()})
        assert runner.world.state_hash()==before and path.stat().st_size==size
        result=runner.advance({runner.world.ids[0]:Action("craft")})
        assert result[0]["reason"]=="unknown_action"
        saved=runner.world.snapshot()
    with JournalRunner.open(path) as resumed:assert resumed.world.snapshot()==saved

def test_corrupt_and_incomplete_records_are_not_silently_discarded(tmp_path):
    path=tmp_path/"world.lwj"
    with JournalRunner.create(path,42,4) as runner:runner.advance({})
    data=path.read_bytes()
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError,match="incomplete"):JournalRunner.open(path)
    assert path.read_bytes()==data[:-1]
    path.write_bytes(data.replace(b'"state_hash":"',b'"state_hash":"f',1))
    with pytest.raises(ValueError,match="integrity"):JournalRunner.open(path)

@pytest.mark.skipif(importlib.util.find_spec("torch") is None,reason="Optional neural dependency")
def test_neural_journal_replay_restores_per_person_decision_state(tmp_path):
    import torch
    from training.learn import LearnedPolicy,RecurrentPolicy,save_artifact
    torch.set_num_threads(1);torch.manual_seed(7)
    artifact=tmp_path/"model.pt";save_artifact(RecurrentPolicy(),artifact,{"test":True})
    first,second=LearnedPolicy(artifact),LearnedPolicy(artifact)
    path=tmp_path/"neural.lwj"
    with JournalRunner.create(path,42,8,policy=first) as runner:
        for _ in range(16):runner.advance_policy()
        memory=first.memory_snapshot();snapshot=runner.world.snapshot()
    with JournalRunner.open(path,second) as resumed:
        assert second.memory_snapshot()==memory and resumed.world.snapshot()==snapshot
        resumed.advance_policy()

def test_failed_persistence_never_exposes_the_proposed_world(tmp_path,monkeypatch):
    import training.journal as journal
    path=tmp_path/"failure.lwj"
    with JournalRunner.create(path,42,4) as runner:
        before=runner.world.snapshot()
        def fail(_):raise OSError("Injected disk flush failure")
        monkeypatch.setattr(journal.os,"fsync",fail)
        with pytest.raises(OSError,match="Injected"):runner.advance({runner.world.ids[0]:Action("depart")})
        assert runner.world.snapshot()==before
        with pytest.raises(RuntimeError,match="reopen"):runner.advance({})
    # The complete line may have reached the filesystem despite failed fsync. Replay
    # determines that outcome; the command is never applied a second time live.
    with JournalRunner.open(path) as resumed:
        assert resumed.world.observe(resumed.world.ids[0])["self"]["group"]==0

def test_process_death_releases_lease_and_incomplete_tail_recovers_explicitly(tmp_path):
    root=Path(__file__).resolve().parents[1];source=tmp_path/"crashed.lwj"
    program="""import os,sys
from training.journal import JournalRunner
from training.policies import HeuristicPolicy
r=JournalRunner.create(sys.argv[1],42,8,policy=HeuristicPolicy())
for _ in range(8):r.advance_policy()
os._exit(23)
"""
    result=subprocess.run([sys.executable,"-c",program,str(source)],cwd=root,check=False)
    assert result.returncode==23
    with JournalRunner.open(source,HeuristicPolicy()) as resumed:
        assert resumed.world.observe(resumed.world.ids[0])["tick"]==8
        snapshot=resumed.world.snapshot()
    interrupted="""import os,sys
f=open(sys.argv[1],'ab');f.write(b'{\"payload\":{\"kind\":\"step');f.flush();os.fsync(f.fileno());os._exit(24)
"""
    result=subprocess.run([sys.executable,"-c",interrupted,str(source)],cwd=root,check=False)
    assert result.returncode==24
    before=source.read_bytes()
    with pytest.raises(ValueError,match="incomplete"):JournalRunner.open(source,HeuristicPolicy())
    target=tmp_path/"recovered.lwj";metadata=recover_incomplete_tail(source,target,HeuristicPolicy())
    assert source.read_bytes()==before and metadata["original_preserved"] and metadata["last_tick"]==8
    with JournalRunner.open(target,HeuristicPolicy()) as recovered:assert recovered.world.snapshot()==snapshot
    with pytest.raises(FileExistsError):recover_incomplete_tail(source,target,HeuristicPolicy())

def test_recovery_never_drops_complete_corruption(tmp_path):
    source=tmp_path/"corrupt.lwj";target=tmp_path/"recovered.lwj"
    with JournalRunner.create(source,42,4) as runner:runner.advance({})
    data=source.read_bytes().replace(b'"state_hash":"',b'"state_hash":"bad',1)+b"partial"
    source.write_bytes(data)
    with pytest.raises(ValueError,match="Complete corrupt"):recover_incomplete_tail(source,target)
    assert not target.exists() and source.read_bytes()==data

def test_compatible_runtime_audit_is_read_only_and_preserves_the_original(tmp_path):
    import hashlib
    from training.checkpoint import canonical
    path=tmp_path/"legacy.lwj"
    with JournalRunner.create(path,42,8,policy=HeuristicPolicy(),rules_version=1) as runner:
        for _ in range(8):runner.advance_policy()
        expected=runner.world.snapshot()
    envelopes=[json.loads(line) for line in path.read_bytes().splitlines()]
    envelopes[0]["payload"]["native_sha256"]="0"*64
    previous="0"*64;lines=[]
    for envelope in envelopes:
        payload=envelope["payload"];digest=hashlib.sha256(previous.encode()+canonical(payload)).hexdigest()
        lines.append(canonical({"previous":previous,"sha256":digest,"payload":payload})+b"\n");previous=digest
    path.write_bytes(b"".join(lines));original=path.read_bytes()
    with pytest.raises(ValueError,match="runtime differs"):JournalRunner.open(path,HeuristicPolicy())
    with JournalRunner.open(path,HeuristicPolicy(),allow_runtime_migration=True) as audited:
        assert audited.world.snapshot()==expected
        with pytest.raises(RuntimeError,match="read-only"):audited.advance_policy()
    assert path.read_bytes()==original

def test_new_journal_from_complete_checkpoint_preserves_state_and_can_continue(tmp_path):
    checkpoint=tmp_path/"checkpoint"
    with JournalRunner.create(tmp_path/"first.lwj",42,8,policy=HeuristicPolicy()) as original:
        for _ in range(8):original.advance_policy()
        original.world.migration_info={"source_journal_sha256":"a"*64,"from_rules":1,"to_rules":2}
        save_checkpoint(original.world,checkpoint,original.policy);expected=original.world.snapshot()
    path=tmp_path/"continued.lwj"
    with JournalRunner.from_checkpoint(path,checkpoint,HeuristicPolicy()) as continued:
        assert continued.world.snapshot()==expected
        continued.advance_policy();final=continued.world.snapshot()
    with JournalRunner.open(path,HeuristicPolicy()) as verified:
        assert verified.world.snapshot()==final
        assert verified.world.migration_info["source_journal_sha256"]=="a"*64
        save_checkpoint(verified.world,tmp_path/"next",verified.policy)
        assert json.loads((tmp_path/"next"/"manifest.json").read_bytes())["migration"]["source_journal_sha256"]=="a"*64

def test_replay_observers_cannot_modify_commands_or_authoritative_observations(tmp_path):
    path=tmp_path/"trace.lwj"
    with JournalRunner.create(path,42,4) as runner:
        runner.advance({runner.world.ids[0]:Action("depart")});expected=runner.world.snapshot()
    def trace(payload,observations):
        if payload["kind"]=="step":payload["actions"].clear()
        for observation in observations.values():observation["self"]["inventory"][0]=999
    with JournalRunner.open(path,trace=trace) as verified:assert verified.world.snapshot()==expected
