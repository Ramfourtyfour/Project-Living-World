# Fixed-seed causal journals

These synthetic journals were selected by protocol (first sealed seed 20000), not
by an impressive outcome. They contain all committed actions/results and state
hashes; the learned log also preserves every individual's recurrent state. No real
player data or account information is included.

These are legacy rules-1 logs. The current engine preserves their state/action
semantics. Explicit compatible audit verifies every recorded result/hash using the
new binary, remains read-only, and does not restore or execute a neural controller.
Decompress into this directory:

```text
python -m gzip -d examples/sealed-20000-heuristic.lwj.gz
python -m gzip -d examples/sealed-20000-native.lwj.gz
python -S tools/journal.py --output examples/sealed-20000-heuristic.lwj --verify-only --audit-compatible
python -S tools/journal.py --output examples/sealed-20000-native.lwj --verify-only --audit-compatible
```

Both logs have 32 initial people, scarce scenario 1 and 512 committed ticks. Final
hashes are `2470712abf944af1` (heuristic) and `b21db7d485dd97fb` (native learned).
The learned policy had 24 survivors versus 17 for the heuristic on this one seed;
the aggregate report, not this single example, is the comparison evidence.

Resuming requires a verified migration and the matching controller, rather than
turning off the runtime check. See `docs/social-contracts.md` and
`tools/migrate_journal.py`. The original logs remain untouched.

`tools/analyze_journal.py` reconstructs local sightings and counts successful
food/water extraction from sources absent from each person's initial observation.
Eight learned people harvested 55 such units; eight heuristic people harvested 93.
The corresponding individual reports are in `benchmarks/`.
