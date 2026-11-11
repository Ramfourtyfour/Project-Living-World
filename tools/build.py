"""Dependency-free C++20 build; CMake is also supported."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true")
    p.add_argument("--build-dir", default="build", help="Use a separate directory if a running process has loaded the current DLL")
    args = p.parse_args()
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        fallback = Path("C:/msys64/ucrt64/bin/g++.exe")
        compiler = str(fallback) if fallback.exists() else None
    if not compiler:
        sys.exit("Install a C++20 compiler or use CMake with Visual Studio.")
    build = (ROOT / args.build_dir).resolve()
    build.mkdir(parents=True,exist_ok=True)
    windows = sys.platform == "win32"
    lib = build / ("livingworld.dll" if windows else "liblivingworld.dylib" if sys.platform == "darwin" else "liblivingworld.so")
    common = [compiler, "-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-O0" if args.debug else "-O2", "-g", "-Isim"]
    if windows:
        # Distribute one DLL, without depending on the user's compiler PATH.
        common += ["-static-libgcc", "-static-libstdc++", "-static"]
    else:
        common += ["-fPIC"]
    subprocess.run(common + ["-shared", "sim/world.cpp", "sim/api.cpp", "-o", str(lib)], cwd=ROOT, check=True)
    native = build / ("native_tests.exe" if windows else "native_tests")
    subprocess.run(common + ["tests/native.cpp", "sim/world.cpp", "-o", str(native)], cwd=ROOT, check=True)
    subprocess.run([str(native)], cwd=ROOT, check=True)
    print(f"Built {lib}")

if __name__ == "__main__":
    main()
