# Durable authority and complete controller checkpoints

`JournalRunner` is a trusted local, single-writer authority, not the network server.
It owns its world; use `runner.world` after each committed tick because the prior
native handle is retired. Human/manual actions and policy actions use the same
validated native API. Neural weights remain fixed, with hidden state per person.

## Tick commit

1. Read the current world snapshot and controller state.
2. Run proposed actions on a native snapshot copy with explicit request IDs.
3. Append a canonical JSON record containing all actions, results, state hash and
   controller memory to a hash-linked private journal, flush it and call `fsync`.
4. Only after persistence succeeds, publish the proposed world as authoritative.

A failed validation leaves the old world untouched. A failed append/flush leaves
the old live world untouched and disables the runner until reopen/replay determines
which complete records reached disk. A complete uncertain record can therefore be
recovered exactly once; the native request sequence prevents later replay of the
same transfer. The journal records ordinary rejected action results too.

One OS file lease prevents two writers using the same file. The kernel releases
the lease on process death; there is no guessed stale lock-file removal. Opening
streams and reconstructs the exact world from its initial snapshot and every
accepted journal commit, then restores the matching controller's latest recurrent
state. Controller version changes require a new journal/checkpoint. New worlds
require fresh controller memory so different worlds never share personal memories.

The journal is **private authority data**. It includes actions, private results and
neural memory; it must never be sent wholesale to ordinary clients. Remote clients
will receive filtered observations only when the network milestone is implemented.

Read-only compatible audit can replay an older engine's recorded actions only if
every action result and world hash still matches. It cannot append or run a
controller. Full resume uses a verified migration into a new checkpoint/journal;
`JournalRunner.from_checkpoint` preserves the validated world/controller bundle.
Trace callbacks receive copies and cannot mutate replayed commands or native state.

## Incomplete writes and explicit recovery

Opening refuses an incomplete last line, a broken hash chain, native replay
divergence, unknown contract, or different native runtime. It never silently drops
records to make a corrupt save appear valid. Recovery of an interrupted final write
requires a named new file:

```text
python tools/journal.py --output runs/world.lwj --recover-to runs/recovered.lwj
```

Recovery leases the source, checks every complete record, copies only the complete
prefix into a new file, replays it and writes a recovery report. The original and
its incomplete bytes are preserved. It refuses to discard a **complete** corrupt
record and refuses to overwrite either source or target. Neural journals require
the same `--policy` artifact when reopening or recovering.

## Complete checkpoint bundles

`save_checkpoint(world, directory, policy)` creates an immutable new directory with
the world snapshot and controller state. Each file is flushed/fsynced first; an
atomic final rename publishes `manifest.json`. No loader accepts a bundle without
that marker. The manifest binds file SHA-256 hashes, ordered individual IDs, world
hash and exact native binary identity. Neural state binds the full policy SHA-256;
loading requires the matching policy and rejects unknown individual references.

An interrupted bundle does not replace any earlier checkpoint. Different native
builds currently require explicit migration, rather than silently weakening the
same-build replay guarantee. FNV world hashes and SHA-256 checks detect accidental
corruption; they do not authenticate an attacker rewriting trusted local files.
File flushing and process-interruption tests do not establish guarantees about
hardware failures or every filesystem's power-loss behavior.

## Run, resume and verify

```text
python tools/journal.py --output runs/world.lwj --ticks 128 --checkpoint runs/checkpoint-128
python tools/journal.py --output runs/world.lwj --resume --ticks 32
python tools/journal.py --output runs/world.lwj --verify-only
```

The historical rules-1 32-person, 128-tick run retained all individuals, with p95 durable tick
latency 17.54 ms on this machine. After 32 further ticks, reopen/replay reproduced
the exact tick-160 hash `82d7a527ad52f82f`, with all 32 alive and conservation true.
These are headless authority measurements, not a rendering or network benchmark.
See `benchmarks/durable-*.json` for raw measurements.

The rules-2 social baseline also completed a 64-person, 512-tick durable stress
run with all 64 alive and resources conserved. Its measured p95 durable tick was
128.04 ms with other desktop activity; this is a non-isolated headless profile,
not a client frame-time claim. See `social-64-profile.json`.

The rules-3 32-person social baseline reached tick 512 with all people alive and
conservation true. Reopen reproduced `bb0215024021a6a9`, including facing, sighting
memories, anonymous hearing and night phase. The measured non-isolated p95 durable
tick was 44.82 ms. This is a headless authority measurement, not a render/network
frame time. `examples/perception-42-social.lwj.gz` preserves the complete log;
see `benchmarks/perception-durable.json` and `perception-replay.json`.
