# Fixed-seed causal journals

These synthetic journals were selected by protocol (first sealed seed 20000), not
by an impressive outcome. They contain all committed actions/results and state
hashes; the learned log also preserves every individual's recurrent state. No real
player data or account information is included.

The exact Windows simulation/policy binaries in the v0.2 package are required for
the currently strict replay check. Different builds require an explicit migration;
cross-platform replay is not claimed. Decompress into this directory:

```text
python -m gzip -d examples/sealed-20000-heuristic.lwj.gz
python -m gzip -d examples/sealed-20000-native.lwj.gz
python -S tools/journal.py --output examples/sealed-20000-heuristic.lwj --verify-only
python -S tools/journal.py --output examples/sealed-20000-native.lwj --native-policy inference/artifacts/curriculum-v3.lwgru --verify-only
```

Both logs have 32 initial people, scarce scenario 1 and 512 committed ticks. Final
hashes are `2470712abf944af1` (heuristic) and `b21db7d485dd97fb` (native learned).
The learned policy had 24 survivors versus 17 for the heuristic on this one seed;
the aggregate report, not this single example, is the comparison evidence.

`tools/analyze_journal.py` reconstructs local sightings and counts successful
food/water extraction from sources absent from each person's initial observation.
Eight learned people harvested 55 such units; eight heuristic people harvested 93.
The corresponding individual reports are in `benchmarks/`.
