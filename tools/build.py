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
    p.add_argument("--inference-only",action="store_true",help="Build the separate native policy runtime without replacing a loaded simulation DLL")
    p.add_argument("--simulation-only",action="store_true",help="Preserve the versioned policy runtime when building simulation rules")
    p.add_argument("--social-inference-only",action="store_true",help="Build only the separate social runtime; preserve simulation and frozen survival DLLs")
    p.add_argument("--social-v2-only",action="store_true",help="Build only the version-2 social runtime")
    args = p.parse_args()
    if sum((args.inference_only,args.simulation_only,args.social_inference_only,args.social_v2_only))>1:raise SystemExit("Choose one component or build all")
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
    if not args.inference_only and not args.social_inference_only and not args.social_v2_only:
        subprocess.run(common + ["-shared", "sim/world.cpp", "sim/social.cpp", "sim/perception.cpp", "sim/api.cpp", "-o", str(lib)], cwd=ROOT, check=True)
        native = build / ("native_tests.exe" if windows else "native_tests")
        subprocess.run(common + ["tests/native.cpp", "sim/world.cpp", "sim/social.cpp", "sim/perception.cpp", "-o", str(native)], cwd=ROOT, check=True)
        subprocess.run([str(native)], cwd=ROOT, check=True)
        print(f"Built {lib}")
    if args.simulation_only:return
    if not args.social_inference_only and not args.social_v2_only:
        policy=build/("livingworld_policy.dll" if windows else "liblivingworld_policy.dylib" if sys.platform=="darwin" else "liblivingworld_policy.so")
        subprocess.run(common+["-Iinference","-shared","inference/runtime.cpp","-o",str(policy)],cwd=ROOT,check=True)
        test=build/("inference_tests.exe" if windows else "inference_tests")
        subprocess.run(common+["-Iinference","tests/inference_native.cpp","inference/runtime.cpp","-o",str(test)],cwd=ROOT,check=True)
        subprocess.run([str(test)],cwd=ROOT,check=True)
        print(f"Built {policy}")
    for version in ([2] if args.social_v2_only else [1] if args.social_inference_only else [1,2]):
        stem="livingworld_social"+("_v2" if version==2 else "")
        social=build/(stem+".dll" if windows else "lib"+stem+(".dylib" if sys.platform=="darwin" else ".so"))
        options=common+["-Iinference",f"-DLW_SOCIAL_VERSION={version}"]
        subprocess.run(options+["-shared","inference/social_runtime.cpp","-o",str(social)],cwd=ROOT,check=True)
        test=build/(f"social_v{version}_tests.exe" if windows else f"social_v{version}_tests")
        subprocess.run(options+["tests/social_inference_native.cpp","inference/social_runtime.cpp","-o",str(test)],cwd=ROOT,check=True)
        subprocess.run([str(test)],cwd=ROOT,check=True);print(f"Built {social}")

if __name__ == "__main__":
    main()
