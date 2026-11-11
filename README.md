# Project Living World

An original standalone survival simulation for a future first-person co-op RPG
inhabited by independent, locally informed agents.

**Current version: 0.2.0 — native simulation and learned-agent prototype.** The full game described
in [the master plan](docs/master-plan.md) is **not complete**. This repository
implements the initial simulation/training work package and reports its results.
There is no 3D client, network co-op, construction, combat or population magic yet.
No code or assets from Shattered Pixel Dungeon / Shattered PD Multiplayer are used.

## Try it on Windows

The Windows package needs Python 3.11+ with Tkinter. It contains both native DLLs
and the pretrained policy; live play needs no PyTorch or NumPy. A source checkout
also needs a C++20 compiler (MSYS2 UCRT64 GCC works).

```powershell
python tools/build.py
python -S tools/inspector.py --native-policy inference/artifacts/curriculum-v3.lwgru
```

Or double-click **Launch Learned Inspector.cmd**. **Launch Inspector.cmd** uses the
heuristic baseline for comparison. The inspector is a desktop diagnostic
viewer, not the first-person game. It opens paused with 32 individuals across four
initial communities. Run, step, inspect people, and toggle manual control. In manual
mode click terrain to move; G gathers the nearest visible node, E eats, D drinks,
R rests and L leaves the selected individual's agreement. Save/load retains the
world. Learned-mode saves include every individual's recurrent memory in a complete
checkpoint folder. The separately labelled oracle toggle exposes global state for debugging.

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
python tools/journal.py --output runs/world.lwj --ticks 128 --checkpoint runs/checkpoint-128
python tools/journal.py --output runs/world.lwj --resume --ticks 32
```

CMake alternative (including Visual Studio/MSVC):

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Set `LIVINGWORLD_LIBRARY` to an absolute library path if using a custom build
directory, and `LIVINGWORLD_POLICY_LIBRARY` for the separate policy library.
Windows MSVC and Ubuntu GCC builds/tests have passed GitHub CI; the local Windows
GCC run includes all 44 tests. See [the verified platform matrix](docs/platforms.md)
for exact scope, versions and optional-test skips.

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
- Complete world/neural-memory checkpoint bundles and a single-writer durable
  action journal with explicit, original-preserving interrupted-write recovery.
- Heuristic/random baselines; a simultaneous Python environment; versioned GRU48/
  GRU64 policies with individual hidden states; balanced imitation and recurrent PPO.
- A checked float32 export and batched C++ recurrent runtime, with no deep-learning
  packages needed in live execution. Numeric conformance and individual memory tested.
- Policy hashes and manifests, held-out paired evaluation, latency/throughput reports.

## Learning experiments

```powershell
python -m training.learn --seed 17 --updates 16 --bc-epochs 16 --output runs/my-policy.pt
python -m training.evaluate --policy runs/my-policy.pt --seeds 8 --ticks 512 --output runs/my-evaluation.json
```

The third predeclared curriculum passed its small survival/exploration pilot gate.
On 16 untouched test seeds (32 individuals, 512 ticks), native learned inference
achieved **57.42% scarce-world survival versus 46.29% for the heuristic**, with
broader map coverage. Both retained 100% survival on normal worlds. The paired
survival gain was 11.13 percentage points; a world-seed bootstrap interval was
6.25–16.21 points. This is bounded evidence for these two scene distributions.

The fixed first-seed causal trace verifies 55 units harvested from newly discovered
survival sources by eight learned agents. The heuristic used eight such agents and
harvested 93 units, so broader coverage does not establish superior discovery
efficiency. Earlier failed runs remain in `benchmarks/`; see
[research results](docs/rl-training.md) and [native inference](docs/native-inference.md).

Social choices, emergent society and the full game remain unproven. Current learned
policies have eight survival/exploration intents and do not choose social actions.
Live neural weights stay fixed; individual recurrent memory can change.

## Code map

| Directory | Role |
| --- | --- |
| `sim/` | C++ state, validation, perception, persistence and C ABI |
| `training/` | ctypes binding, simultaneous environment, baselines, recurrent PPO/evaluation |
| `inference/` | Checked native model export, C++ recurrent runtime and dependency-free binding |
| `inference/artifacts/` | Experimental policies and versioned manifests |
| `tools/` | Build, desktop inspector, full rollout/replay and profiling |
| `tests/` | Native checks and Python behavior/adversarial tests |
| `benchmarks/` | Actual machine-readable measurements and evaluation results |
| `docs/` | Contracts, acceptance evidence, full plan and milestone status |

Read [simulation contracts](docs/simulation-contracts.md),
[acceptance tests](docs/acceptance-tests.md), [milestones](docs/milestones.md) and
[licensing policy](docs/licensing.md) before extending the engine. The full plan
is a design reference; task instructions in it do not authorize external actions.

See [durable authority](docs/durable-authority.md) for run/resume/recovery commands
and the precise persistence guarantees. Training now versions its local feature
encoder and reward configuration; old policy artifacts retain their old encoder.
