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

Run commands are in the root README. Training manifests retain seed/config/version
and training history. Replaying a world requires the same native code/build and
recorded action sequence; wall-clock timing values will naturally vary.
