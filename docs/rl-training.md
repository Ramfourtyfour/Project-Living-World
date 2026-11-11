# Learning experiments and claims

The first experiment trains a shared 48-unit GRU actor/critic. Every individual
has its own recurrent vector and receives only actor-local structured features.
Intent execution is deterministic ordinary navigation/interactions. It cannot
complete a harvest remotely or see hidden resources.

The pilot uses behavior-cloning demonstrations from `heuristic-v1`, followed by
clipped PPO with four recurrent optimization passes, gamma 0.99, lambda 0.95,
gradient clipping, value loss and a categorical masked action distribution.
[PyTorch's distribution interface](https://docs.pytorch.org/docs/stable/distributions.html)
provides sampling, log probabilities and entropy. This is a local pilot, not a
validated MAPPO implementation or league/self-play system.

Training world seeds are below 9000. Evaluation seeds begin at 10000, with the same
eight held-out worlds for every policy. No evaluation scores feed the optimizer.
Metrics include survival, lifetime, explored 10 m cells, invalid action rate,
conservation and p95 batched inference latency. Exploration cells are a coarse
proxy, not proof of useful discovery. Learning reward decomposes survival, health,
need reduction, rest, safe new-cell exploration and invalid-action penalties.

## Preserved first experiment

Configuration: seed 17, 16 individuals, 256-tick training episodes, 8 demonstration
episodes, 16 cloning epochs, 16 PPO updates. Artifacts: `pilot.pt` and `pilot-bc.pt`.
Evaluation: 32 individuals, eight held-out seeds, 512-tick horizon.

| Policy | Mean final survival | Mean lifetime | Mean explored cells/person |
| --- | ---: | ---: | ---: |
| Random | 0% | 133.24 ticks | 3.97 |
| Heuristic | 100% | 512 ticks | 30.58 |
| Cloning warm start | 0% | 96.43 ticks | 17.55 |
| Cloning + PPO | 0% | 96.43 ticks | 18.25 |
| Balanced cloning, 64 epochs | 100% | 512 ticks | 29.90 |

The learning gate failed. The initially unbalanced training data encouraged
frequent exploration while neglecting rare consumption actions. This is a
diagnostic hypothesis based on the behavior and class distribution, not a proven
causal ablation. A subsequent experiment adds weighted cloning and lowers the
policy learning rate. Weighted cloning recovered full survival on the same eight
held-out seeds, but exploration remained slightly below the heuristic. Both
original failures remain available for comparison. This recovery is imitation
learning, not evidence that reinforcement learning beat the teacher.

The current normal-world heuristic reaches the survival ceiling. Strictly greater
survival is impossible on this distribution; a meaningful next gate must retain
these easy seeds and add separately declared scarcity/shock evaluation scenarios.
Do not change the metric or replace the baseline after inspecting a score to
manufacture success. The second curriculum configuration and acceptance test were
predeclared in `benchmarks/curriculum-v2-protocol.md` before scarcity evaluation.

## Predeclared scarcity experiment

Starting from balanced cloning, 32 PPO updates on 32-person scarce worlds produced
205,166 training agent steps in 69.96 seconds locally. The same eight held-out seeds
were then evaluated in normal and scarce scenarios for 512 ticks.

| Scenario / policy | Mean final survival | Mean lifetime | Explored cells/person |
| --- | ---: | ---: | ---: |
| Normal heuristic | 100% | 512 | 30.58 |
| Normal learned | 100% | 512 | 32.45 |
| Scarce heuristic | 46.09% | 349.15 | 31.02 |
| Scarce learned | 33.20% | 303.12 | 28.54 |

The learned policy improved normal exploration while retaining survival, but
underperformed the heuristic under scarcity. The **predeclared gate failed**.
The complete result is `benchmarks/curriculum-v2-evaluation.json`. This is a useful
negative result, not permission to claim the full game or successful social MARL.
The next research increment should inspect resource waste and scarcity trajectories,
extend training horizons, and run declared reward/memory ablations.

No social learning, emergent civilization, long-run economy stability, independent
monster behavior or online neural weight learning has been demonstrated. Negative
results remain preserved as evidence rather than overwritten by subsequent gains.

## Third curriculum: passed bounded pilot

The development diagnosis found fatigue damage and deaths with owned supplies in
the learned policy; food waste alone did not explain the failures. The third
protocol was declared before its evaluation. It added 38 local features and a
64-unit GRU, alternating normal/scarce cloning demonstrations, a 512-tick PPO
horizon, a normal/scarce training mixture and an explicit resource-waste reward
component. This combined intervention does not identify the contribution of any
single change. Architecture and reward versions retain legacy artifact behavior.

The final run used 96 PPO updates and 1,266,864 agent steps. Exact configuration,
source file hashes, a frozen source ZIP, native binary SHA-256 and training history
are retained with `curriculum-v3.pt`. The live policy is exported unchanged to C++.

| Test scenario / policy | Mean final survival | Mean lifetime | Explored cells/person |
| --- | ---: | ---: | ---: |
| Normal heuristic | 100% | 512 | 31.29 |
| Normal learned, native | 100% | 512 | 40.35 |
| Scarce heuristic | 46.29% | 350.07 | 31.47 |
| Scarce learned, native | 57.42% | 399.77 | 39.07 |

These are 16 sealed seeds 20000..20015, separate from optimization, development
and the repeatedly inspected validation seeds 10000..10007. The frozen policy
passed validation and the sealed comparison. Resource conservation holds in every
sealed episode, including random-policy failures. It won survival on 13 world
seeds, tied two and lost one. A paired world-seed bootstrap interval for the 11.13
point survival gain was [6.25, 16.21] points; agents within a world are correlated
and are not treated as independent samples.

The first sealed seed was fixed in advance for causal tracing. Eight learned
agents extracted 55 food/water units from sites outside their initial sight. The
heuristic also had eight such users and extracted 93 units. This establishes real
use of discoveries while limiting claims about exploration efficiency. Full
compressed journals and individual reports are retained; no best-seed selection.

The pilot stop/go gate is now passed for the declared simple survival worlds.
This does not pass social training, richer environment, society, 3D, co-op or
release gates. Memory/reward/relationship ablations and broader league experiments
remain required as those capabilities are added. See `curriculum-v3-summary.json`
and the predeclared protocol for the machine-readable audit and limits.
