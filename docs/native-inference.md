# Native recurrent policy runtime

The live runtime is an original small C++20 implementation of the two declared
single-layer GRU architectures. It follows the reset/update/new gate equations
and weight order in the [official PyTorch GRU contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html),
including reset multiplication after the hidden linear transform. It is not a
general model importer or an implementation of a new learning algorithm.

The master plan suggested evaluating ONNX Runtime/libtorch. For this fixed small
prototype, a bounded native kernel keeps deployment independent of those large
runtime dependencies. General architectures and production batching/acceleration
remain future work. The C ABI can be hosted by a native server; Python currently
packs local features and executes ordinary world intents around the native call.

## Checked model contract

`inference.export` loads the source artifact with PyTorch weights-only loading,
verifies its SHA-256, and writes `lwgru-v1`: an 8-byte magic, five little-endian
uint32 values (format version, input width, hidden width, action count, tensor
value count), then the declared float32 tensors. The manifest records source/exported
hashes, architecture and feature version. Only input/hidden pairs 22/48 and 38/64
with eight outputs are accepted. Weights must be finite and bounded by 1000.
Truncated, oversized, trailing, dimension-mismatched and corrupt data is rejected.

The C++ API accepts a batch of at most 256 distinct positive IDs, finite bounded
features and nonempty action masks. Each ID owns independent hidden state.
Validation and evaluation precede the atomic state-map swap. At most 4096 tracked
IDs are supported per controller instance. Restoring recurrent memory validates
the entire replacement before committing; runtime and model hashes bind saves.
Handles are single-threaded, with thread-local error text. UTF-8 model paths are
supported. No user data or policy memory is sent to an external inference service.

## Conformance evidence

Random reference-model tests cover both architectures, reordered/subset batches,
state restoration and masks. The actual trained policy was then compared on
512-tick normal/scarce development trajectories (seeds 9010/9011): 28,966 decisions,
zero intent disagreement, max logit absolute error 0.00001383 and max hidden-state
error 0.00000480. The declared tolerance was 0.0001. This is numerical conformance,
not bitwise cross-runtime identity.

Native batch p95 was about 1.53–1.57 ms versus PyTorch's 0.85–0.88 ms in that local
comparison. The small scalar native kernel is **slower** here; its current benefit
is dependency-free deployment and explicit state ownership. It is not a speedup
claim. The full sealed evaluation subsequently ran the native policy without site
packages, and dependency-free world/checkpoint execution is integration-tested.

## Use

```text
python tools/build.py --inference-only
python -m inference.export inference/artifacts/curriculum-v3.pt runs/policy.lwgru
python -S tools/inspector.py --native-policy inference/artifacts/curriculum-v3.lwgru
python -S tools/journal.py --native-policy inference/artifacts/curriculum-v3.lwgru --output runs/native.lwj --ticks 128
```

Training/export needs PyTorch/NumPy. The exported runtime, inspector and durable
world loop require only Python's standard library and the two native libraries.
The Windows package contains those libraries and the selected frozen model.

## Experimental social runtime

`social_runtime.cpp` implements the independent GRU64 context plus shared
candidate-scoring MLP. Build versions 1 and 2 into separate libraries with
candidate widths 52 and 64. `lwslate-v1/v2` use eight-byte magic, six uint32 fields
and respectively 37,258/38,026 bounded float32 parameters. Neither format changes
the existing `lwgru-v1` runtime or frozen survival weights.

Each batch has distinct individual IDs, 64 context inputs and variable candidate
slices (1–128 per person), with at most 256 people and 4,096 remembered identities.
Invalid offsets, masks, features and memory are rejected before memory commits.
Manifests bind observation/proposal decoder sources as well as model and runtime
hashes. Complete checkpoints retain independent hidden vectors. A real `python -S`
checkpoint continuation test covers v2 without PyTorch or NumPy.

The actual v1 trained model agrees with PyTorch on 4,736 development decisions:
zero choice disagreements, maximum logit error 0.00001526 and hidden error
0.00000036. This is arithmetic conformance, not evidence that its behavior passed
social validation. See `social-learning.md` for behavioral results and v2 commands.
