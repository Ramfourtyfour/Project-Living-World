# Third-party notices

The Windows prototype's native library is compiled with MSYS2 UCRT64 GCC 16.2.0.
It statically links the GCC C++/compiler runtime. The GNU GPL version 3 and GCC
Runtime Library Exception are reproduced in `LICENSES/GCC-COPYING3.txt` and
`LICENSES/GCC-RUNTIME-EXCEPTION.txt`. This does not place original independent
Project Living World source under the GPL. Refer to those upstream terms for the
precise grant and conditions.

MinGW-w64 CRT/runtime notices and the winpthreads notice used by this toolchain
are reproduced in `LICENSES/MINGW-CRT.txt`, `LICENSES/MINGW-W64.txt`,
`LICENSES/MINGW-W64-RUNTIME.txt` and `LICENSES/WINPTHREADS.txt`.

The DLL dynamically imports the Windows system and Universal C Runtime APIs. Those
OS libraries are not distributed in this package. No Python, Tcl/Tk, PyTorch,
NumPy, pytest, Godot, third-party art or audio runtime is bundled. Install external
tools separately under their own licenses. Experimental weights were trained
locally on the synthetic simulation.

This notice covers the prototype components actually packaged; a full game release
requires a renewed dependency/asset audit.
