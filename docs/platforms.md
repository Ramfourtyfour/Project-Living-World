# Platform and code conventions

The initial supported target is desktop Windows/Linux x64. Mac, other CPU
architectures and other device platforms are unverified future targets.

| Environment | Native toolchain | Evidence |
| --- | --- | --- |
| Local Windows x64, Python 3.13.2 | MSYS2 UCRT64 GCC 16.2.0 | Native world/inference checks, 83 Python tests, PyTorch/native numeric conformance, training/evaluation and packaged-binary checks |
| GitHub Windows x64, Python 3.13 | Visual Studio C++20 through CMake | Both native tests, 75 Python tests, 8 optional learning tests skipped, rollout/replay and full 512-tick sensory-reference audit |
| GitHub Ubuntu x64, Python 3.13 | GCC C++20 through CMake | Both native tests, 75 Python tests, 8 optional learning tests skipped, rollout/replay and full 512-tick sensory-reference audit |

The successful v0.4 [Windows/Linux CI run](https://github.com/Ramfourtyfour/Project-Living-World/actions/runs/43120511320)
checked code commit `997033da9b645eb0970ff4eefabc887a78ed6240`. Optional PyTorch
reference/training tests were run locally; the lean CI matrix deliberately
exercises the dependency-free deployment and core contracts. Python 3.11+ is the
language compatibility target; only the Python versions above are verified. Both
CI platforms reproduced the specific Windows-GCC sensory-reference hash
`bb0215024021a6a9`. This is evidence for that trace, not a universal cross-platform
bitwise-replay guarantee.

## Code conventions

Use C++20 with standard-library-only simulation/inference boundaries, stable
integer identities, explicit bounds/validation, typed C ABI structures and
versioned formats. Use four spaces and UTF-8/LF, except Windows command launchers.
Preserve upstream notice texts and intentional Markdown line breaks. `.editorconfig`
and `.gitattributes` record these editing/file-transfer conventions.

Keep authority in the native world, policy execution separate from world mutation,
and neural/controller state distinct per individual. Never use pointer addresses
as persistent IDs. Python application code targets 3.11+ with standard-library-only
live runtime; training dependencies stay optional at deployment. Tests must cover
the claimed scope, including malformed/duplicate input and checkpoint/replay.

CI runs CMake, native tests, Python contract tests and a recorded rollout/replay.
Numerical and behavioral reports name their exact source/policy/native versions.
Same-binary replay and strict migration checks are current guarantees; neither
cross-platform bitwise replay nor an audited release candidate is claimed.
