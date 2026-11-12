# Reproducible measurements

These files contain actual local runs, including failures. They are not target
performance numbers. Tests and recordings are distinct from human play evidence.

- `baseline-evaluation.json`: random/heuristic, eight held-out normal worlds.
- `pilot-evaluation.json`: original cloning+PPO pilot and both baselines; failed.
- `warm-start-evaluation.json`: original unbalanced cloning checkpoint; failed.
- `balanced-warm-start-evaluation.json`: balanced imitation; matched survival,
  slightly lower exploration than the heuristic; expansion gate failed.
- `throughput.json`: 16/32/64 agents, 256 ticks, end-to-end Python + native runtime,
  with Python allocation tracing enabled. Tracing has overhead and excludes C++
  allocations. It is not a native-only throughput, total RAM or 60 FPS claim.
- `curriculum-v2-protocol.md`: predeclared harder curriculum and gate. Evaluation
  output is written only after the complete paired run finishes.
- `scarcity-diagnosis.json`: development-seed action, waste, discovery and death
  accounting. It is neither the optimization seed set nor the sealed final test.
- `curriculum-v3-protocol.md`: predeclared 38-feature recurrent curriculum and
  separate validation/sealed-test partitions. Warm-start validation is retained.
- `durable-authority.json`, `durable-resume.json`, `durable-replay.json`: actual
  fsynced journal runs and exact reopen/replay measurements.
- `curriculum-v3-validation.json`, `curriculum-v3-sealed-test.json`,
  `curriculum-v3-summary.json`: fixed candidate, 8 validation/16 sealed seeds,
  paired uncertainty and the passed bounded pilot gate.
- `native-inference-conformance.json`: actual trained-weight numerical comparison,
  including the native kernel's slower local timing rather than a speedup claim.
- `sealed-20000-*-discovery.json`: verified first-sealed-seed causal exploration
  accounting, including individual reports. Full compressed logs are in `examples/`.
- `social-mechanics.json`: 16 surplus/16 scarce-stock episodes with real acquisition
  and independent declared-heuristic decisions. Not learned cooperation.
- `rules-v2-survival-regression.json`: reused validation seeds under new cache/
  social rules, with separate scope from the frozen rules-1 pilot.
- `legacy-checkpoint-upgrade.json`, `legacy-journal-compatible-audit.json`: actual
  old-state/controller preservation and exact old-journal replay.
- `social-64-profile.json`: non-isolated 64-person durable stress profile under
  rules 2, with all 64 alive at tick 512 and conservation true.
- `perception-development.json`: four development seeds per normal/scarce scenario
  under rules 3, through sunset/night. No new sealed or sensory-learning claim.
- `perception-legacy-replay.json`, `perception-checkpoint-upgrade.json`: exact old
  rules-2 tick-512 replay and rules-2 neural-memory migration into rules 3.
- `perception-durable.json`, `perception-replay.json`: actual 32-person rules-3
  run through tick 512 and exact reopen/replay; the compressed log is in `examples/`.

Run commands are in the root README. Training manifests retain seed/config/version
and training history. Replaying a world requires the same native code/build and
recorded action sequence; wall-clock timing values will naturally vary.
