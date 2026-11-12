# Project Living World

An original standalone survival simulation for a future first-person co-op RPG
inhabited by independent, locally informed agents.

**Current version: 0.5.0 — native survival and experimental learned social prototype.** The full game described
in [the master plan](docs/master-plan.md) is **not complete**. This repository
implements the initial simulation/training work package and reports its results.
There is no 3D client, network co-op, settlement construction, combat or population magic yet.
No code or assets from Shattered Pixel Dungeon / Shattered PD Multiplayer are used.

## Try it on Windows

The Windows package needs Python 3.11+ with Tkinter. It contains both native DLLs
and the pretrained policy; live play needs no PyTorch or NumPy. A source checkout
also needs a C++20 compiler (MSYS2 UCRT64 GCC works).

```powershell
python tools/build.py
python -S tools/inspector.py --native-policy inference/artifacts/curriculum-v3.lwgru
```

Or double-click **Launch Learned Inspector.cmd**. **Launch Social Inspector.cmd**
uses a declared social heuristic; **Launch Inspector.cmd** uses the survival
heuristic. The inspector is a desktop diagnostic
viewer, not the first-person game. It opens paused with 32 individuals across four
initial communities. Run, step, inspect people, and toggle manual control. In manual
mode click terrain to move or right-click to look. Q scans, C calls, Z toggles
crouching; G gathers a visible node, E eats, D drinks, R rests and L leaves the
selected individual's agreement. Save/load retains the
world. Learned-mode saves include every individual's recurrent memory in a complete
checkpoint folder. The separately labelled oracle toggle exposes global state for debugging.

**Social & storage** pauses the lab and lets the selected person make requests,
answer/refuse proposals, report completed work, trade, found/name/leave a group, and deposit/withdraw from
nearby authorized caches. A new cache costs 3 wood and 2 fiber acquired through
ordinary gathering. Requests never command another person's controller. The
social baseline responds from its own needs/resources; the default frozen learned
model handles survival. Experimental learned social controllers now have a separate
`--social-native-policy` inspector option and complete neural-memory saves. Their
broader behavior gates remain open. See [social contracts](docs/social-contracts.md)
and [social learning](docs/social-learning.md).

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
python tools/social_scenarios.py --journals runs/social-scenarios
python tools/rules_regression.py --output runs/rules-v2-regression.json
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
GCC run includes all 97 tests. See [the verified platform matrix](docs/platforms.md)
for exact scope, versions and optional-test skips.

## What runs

- C++20 authoritative world with deterministic generation, stable IDs, quantized
  continuous movement, obstacle collision and one world clock.
- 1–64 individuals (32 by default), independently sampled styles, empty starting
  inventories, four initial agreements, hunger/thirst/fatigue and death.
- Real node extraction/regrowth and checked per-resource conservation on every tick.
- Local sight with obstruction, remembered sightings with age/provenance/confidence,
  private observations and witnessed event journals. Rules 3 adds facing/FOV,
  day/night light, cover, crouching, anonymous approximate hearing and dated person/
  landmark memories. See [perception contracts](docs/perception-contracts.md).
- Validated move, inspect, gather, consume, gift, reciprocal trade, accept/decline,
  rest, depart and invite actions. Invitations need individual acceptance.
- Dynamic named agreements, private directional trust, explicit help/visit/guard
  proposals, offscreen work with local verification/reporting, physical shared caches and immediate access
  revocation on departure. All survive save/load without material duplication.
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

Experimental GRU/slate models now select real social, membership and storage
actions independently. Their versioned native runtimes, recurrent PPO training,
causal task construction and frozen-decision replay are implemented. The first
candidate failed validation; its negative results and corrected comparison remain
documented in [social learning](docs/social-learning.md). This does not replace the
frozen survival pilot or demonstrate whole-world society.

Social generalization, emergent society and the full game remain unproven. Current frozen learned
policies have eight survival/exploration intents and do not choose social actions.
Live neural weights stay fixed; individual recurrent memory can change. The frozen
survival encoder does not yet learn from hearing, light or facing.

The scores above are frozen **rules-1** results. Rules 2 adds caches and social
contracts; its separate compatibility regression uses previously inspected seeds
and is not a new sealed test. The social scenarios validate mechanics and a declared
heuristic, not learned cooperation. Legacy journals can be audited exactly with
`--verify-only --audit-compatible`; safe migration preserves the original file.
Rules 3 adds directional, light/cover-limited sight and anonymous hearing. Its
development checks and behavior report are separate from the frozen learning gate.

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
