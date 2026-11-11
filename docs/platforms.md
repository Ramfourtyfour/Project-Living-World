# Platform and code conventions

The initial supported target is desktop Windows/Linux x64. Mac, other CPU
architectures and other device platforms are unverified future targets.

| Environment | Native toolchain | Evidence |
| --- | --- | --- |
| Local Windows x64, Python 3.13.2 | MSYS2 UCRT64 GCC 16.2.0 | Native world/inference checks, 44 Python tests, PyTorch/native numeric conformance, training/evaluation and packaged-binary checks |
| GitHub Windows x64, Python 3.13 | Visual Studio C++20 through CMake | Both native tests, 36 Python tests, 8 optional learning tests skipped, rollout/replay |
| GitHub Ubuntu x64, Python 3.13 | GCC C++20 through CMake | Both native tests, 36 Python tests, 8 optional learning tests skipped, rollout/replay |

The successful [Windows/Linux CI run](https://github.com/Ramfourtyfour/Project-Living-World/actions/runs/43102704838)
checked code commit `83ef545daad84182203303588dfb62cb99f183dc`. Optional PyTorch
reference/training tests were run locally; the lean CI matrix deliberately
exercises the dependency-free deployment and core contracts. Python 3.11+ is the
language compatibility target; only the Python versions above are verified.

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
