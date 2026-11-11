# Acceptance evidence

`python tools/build.py` compiles the C++20 library and runs native checks.
`python -m pytest -q` runs the Python/native integration suite. Version 0.3 passed
74 local tests including checkpoint/journal interruption, both neural architectures and
dependency-free native inference. Native world/inference checks also pass.
[Windows/Linux CI](https://github.com/Ramfourtyfour/Project-Living-World/actions/runs/43102704838)
passed on commit `83ef545`: each platform passed two native tests, 36 Python tests
and recorded rollout/replay; eight optional learning tests were skipped there.
Those optional tests passed locally. See `platforms.md` for the exact matrix.

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

Additional v0.2 evidence:

- Immutable world/controller bundles reject partial publication, corrupted files,
  malformed metadata, wrong native/runtime versions and unknown memory identities.
- A real subprocess exits without cleanup; reopening replays its fsynced commits
  and acquires the kernel-released writer lease. Incomplete final-write recovery
  preserves the original and refuses to discard complete corruption.
- Injected persistence failure never publishes the proposed live world. The runner
  becomes unavailable until reopen/replay determines the disk outcome.
- C++ GRU outputs, hidden vectors and masked choices match PyTorch within declared
  tolerance under reordered/subset identity batches. Unicode paths, malformed
  models, invalid masks and atomic memory replacement are covered.
- A `python -S` subprocess runs native policy/world/checkpoint operations without
  importing PyTorch or NumPy. The sealed 16-seed evaluation also uses this runtime.
- The third curriculum passed the bounded declared validation/sealed gate. All
  test episodes conserve materials. A fixed-seed causal trace verifies actual
  harvest from newly discovered sources, with its limitations retained.

Additional v0.3 tests cover separate offscreen task progress/requester beliefs,
truthful in-person reports and correction of mistaken expiry beliefs; voluntary
named agreements and invitations; physical material-funded cache construction,
private/stale stock and access revocation; malformed named-wire batches; exact
legacy replay and original-preserving world/controller migration. Mixed randomized
commands replay and conserve through saves. Migration provenance survives continued
journals/checkpoints. Thirty-two varied-seed social baseline scenarios conserve and
replay, with all surplus requests fulfilled and all scarce requests declined.

The Windows social inspector was visually checked using computer-use on 2026-11-11:
default paused local view, complete social dialog, manual food request, next-step
independent refusal visible in the personal journal, and clean close. It ran with
`python -S`, without optional learning dependencies. This is a limited diagnostic
UI check; no first-person or broad human playtest claim follows. First-person play,
settlement construction, population magic, co-op, lag,
cross-platform replay equivalence, sanitizers and release playthrough are pending.
