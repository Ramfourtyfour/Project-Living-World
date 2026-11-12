# Experimental social candidate v2

These weights use genuine reciprocal-shortage tasks and factual own-cost inputs.
`bc` is the imitation checkpoint; `policy` adds 96 recurrent PPO updates. They are
retained for research and an explicitly labelled inspector mode. The broader
social gate remains open; no sealed evaluation has been run. Configuration, source
ZIP and all checkpoint/native hashes are retained. See `docs/social-learning.md`
and `benchmarks/social-v2-*.json` for the complete validation and negative results.

After building both social libraries, run without Torch or NumPy:

```text
python -S tools/inspector.py --social-native-policy inference/artifacts/social-c2-v2/policy.lwslate
```

Normal launchers retain the frozen survival model. The raw training/evaluation
traces are preserved in the local research archive. These artifacts do not prove
emergent society or replace the full master-plan acceptance gates.
