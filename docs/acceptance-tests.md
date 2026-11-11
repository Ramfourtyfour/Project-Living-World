# Acceptance evidence

`python tools/build.py` compiles the C++20 library and runs native checks.
`python -m pytest -q` runs the Python/native integration suite. The initial suite
passed 22 tests before the scarcity curriculum addition; further changes require
rerunning it. CI configuration is provided
for Ubuntu and Windows; unexecuted CI is not a passed platform gate.

Covered integration scenarios:

- Four fixed-seed 32-person save/resume rollouts produce identical hashes even
  when action dictionary insertion order changes.
- Hidden nodes are absent from observations and cannot be remotely gathered;
  other inventories/needs remain private; rejection events are actor-private.
- Reads do not change world state. Resource accounting survives randomized valid
  and malformed commands, deaths and save restoration.
- Duplicate requests and duplicate actor batches cannot apply an action twice.
- Invitations do not automatically join people; individual depart/accept requests
  work across checkpoints; repeated acceptance cannot replay the transaction.
- Reciprocal transfers commit both sides atomically. Spending an offered item
  makes acceptance fail without minting a replacement.
- Corrupted/version-mismatched snapshots are rejected.
- Simultaneous environment keys and truncation behavior match their documented API.
- Full action/observation recordings reproduce exact state hashes.
- On four declared seeds, at least 28 of 32 heuristic individuals survive 256 ticks
  starting with empty inventories. Held-out benchmark files provide broader metrics.

Native checks additionally cover obstruction-limited perception/body motion,
invalid oversized batches without clock mutation, and stale remote memories.

The desktop inspector has been launched. Computer-use window inspection was
unavailable because app approval timed out, so visual/manual play verification is
not claimed. First-person play, construction, population magic, co-op, lag,
cross-platform replay equivalence, sanitizers and release playthrough are pending.
