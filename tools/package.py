"""Package tracked source plus a prebuilt native library for a reviewable prototype."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);args=p.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    destination=output/"Project-Living-World"
    if destination.exists():
        raise SystemExit("Package directory already exists; select a new output directory to preserve prior deliverables")
    names=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode("utf-8").split("\0")
    files=[ROOT/name for name in names if name]
    if not files:
        raise SystemExit("Stage the reviewed source files first with git add")
    suffix="livingworld.dll" if sys.platform=="win32" else "liblivingworld.dylib" if sys.platform=="darwin" else "liblivingworld.so"
    native=ROOT/"build"/suffix
    if not native.is_file():raise SystemExit("Build the native library first")
    files.append(native)
    policy_name="livingworld_policy.dll" if sys.platform=="win32" else "liblivingworld_policy.dylib" if sys.platform=="darwin" else "liblivingworld_policy.so"
    policy_library=ROOT/"build"/policy_name
    if not policy_library.is_file():raise SystemExit("Build the native policy library first")
    files.append(policy_library)
    manifest={"version":1,"kind":"headless-prototype","full_master_plan_complete":False,"files":{}}
    for source in files:
        relative=source.relative_to(ROOT)
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        manifest["files"][relative.as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
    (destination/"PACKAGE-MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    archive=output/"Project-Living-World-Prototype-Windows.zip"
    if archive.exists():raise SystemExit("Archive already exists; preserve the previous deliverable")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zipped:
        for source in sorted(destination.rglob("*")):
            if source.is_file():zipped.write(source,source.relative_to(output))
    print(json.dumps({"directory":str(destination),"archive":str(archive),"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"bytes":archive.stat().st_size},indent=2))

if __name__=="__main__":main()
