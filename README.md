# Project Living World

An original standalone survival simulation for a future first-person co-op RPG
inhabited by independent, locally informed agents.

**Current version: 0.1.0 — headless research prototype.** The full game described
in [the master plan](docs/master-plan.md) is **not complete**. This repository
implements the initial simulation/training work package and reports its results.
There is no 3D client, network co-op, construction, combat or population magic yet.
No code or assets from Shattered Pixel Dungeon / Shattered PD Multiplayer are used.

## Try it on Windows

Requires Python 3.11+ with Tkinter, and a C++20 compiler (MSYS2 UCRT64 GCC works).
The downloadable Windows prototype includes the prebuilt native library.

```powershell
python tools/build.py
python tools/inspector.py
```

Or double-click **Launch Inspector.cmd**. The inspector is a desktop diagnostic
viewer, not the first-person game. It opens paused with 32 individuals across four
initial communities. Run, step, inspect people, and toggle manual control. In manual
mode click terrain to move; G gathers the nearest visible node, E eats, D drinks,
R rests and L leaves the selected individual's agreement. Save/load retains the
world. The separately labelled oracle toggle exposes global state for debugging.

## Build and verify

The simulation and Python binding have **no third-party runtime dependencies**.
Tests require pytest; learning additionally requires NumPy and PyTorch. Verified
local package versions are pinned in `requirements.txt`.

```powershell
python tools/build.py
python -m pip install -r requirements.txt
python -m pytest -q
python tools/rollout.py --seed 42 --ticks 256 --output runs/demo.jsonl
python tools/rollout.py --replay runs/demo.jsonl
python tools/benchmark.py
```

CMake alternative (including Visual Studio/MSVC):

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Set `LIVINGWORLD_LIBRARY` to an absolute library path if using a custom build
directory. Linux GCC/Clang and Windows MSVC CI configurations are provided;
only the local Windows GCC build has been executed during initial authoring.

## What runs

- C++20 authoritative world with deterministic generation, stable IDs, quantized
  continuous movement, obstacle collision and one world clock.
- 1–64 individuals (32 by default), independently sampled styles, empty starting
  inventories, four initial agreements, hunger/thirst/fatigue and death.
- Real node extraction/regrowth and checked per-resource conservation on every tick.
- Local sight with obstruction, remembered sightings with age/provenance/confidence,
  private observations and witnessed event journals.
- Validated move, inspect, gather, consume, gift, reciprocal trade, accept/decline,
  rest, depart and invite actions. Invitations need individual acceptance.
- Versioned checked snapshots, request deduplication, atomic disk save, pending
  offers, full action/observation JSONL recordings and exact same-build replay.
- Heuristic/random baselines; a simultaneous Python environment; shared GRU48
  policy with individual hidden states; behavior-cloning warm start and PPO pilot.
- Policy hashes and manifests, held-out paired evaluation, latency/throughput reports.

## Learning experiments

```powershell
python -m training.learn --seed 17 --updates 16 --bc-epochs 16
python -m training.evaluate --seeds 8 --ticks 512
```

The first pilot **failed** the learning gate: all learned agents died on eight
held-out worlds; the heuristic kept all 32 alive for 512 ticks. Both random and
learned failures are retained in `benchmarks/`. A policy file is an experimental
artifact, not proof of emergent intelligence. The inspector defaults to the
verified heuristic. A subsequent balanced imitation run matched 100% survival,
but did not exceed the heuristic in exploration. See
[research results](docs/rl-training.md) for subsequent runs.

Expansion requires stronger evidence under the master plan's stop/go criterion.
Successful membership/transfer tests verify mechanics, not learned cooperation.
Live models do not change neural weights; individual recurrent memory can change.

## Code map

| Directory | Role |
| --- | --- |
| `sim/` | C++ state, validation, perception, persistence and C ABI |
| `training/` | ctypes binding, simultaneous environment, baselines, recurrent PPO/evaluation |
| `inference/artifacts/` | Experimental policies and versioned manifests |
| `tools/` | Build, desktop inspector, full rollout/replay and profiling |
| `tests/` | Native checks and Python behavior/adversarial tests |
| `benchmarks/` | Actual machine-readable measurements and evaluation results |
| `docs/` | Contracts, acceptance evidence, full plan and milestone status |

Read [simulation contracts](docs/simulation-contracts.md),
[acceptance tests](docs/acceptance-tests.md), [milestones](docs/milestones.md) and
[licensing policy](docs/licensing.md) before extending the engine. The full plan
is a design reference; task instructions in it do not authorize external actions.
