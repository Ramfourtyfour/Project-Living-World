"""Explicit version dispatch for research tools; never import Torch for native use."""
from functools import partial
import importlib
import json
from pathlib import Path


def proposal_version(path,native=False):
    path=Path(path)
    manifest=path.with_name(path.name+".json") if native else path.with_suffix(".json")
    if manifest.stat().st_size>1024*1024:raise ValueError("Oversized social manifest")
    data=json.loads(manifest.read_bytes());version=data.get("proposal_version")
    if type(version) is not int or version not in (1,2):raise ValueError("Unsupported proposal version")
    if data.get("architecture")!=f"social-gru64-slate-v{version}":raise ValueError("Mismatched social architecture")
    return version


def learned_factory(path,*,native=False,memory=True,relationships=True):
    version=proposal_version(path,native)
    module=("inference.social_runtime" if native else "training.social_model")+("_v2" if version==2 else "")
    if not native:
        import torch
        torch.set_num_threads(1)
    kind=getattr(importlib.import_module(module),"NativeSocialPolicy" if native else "LearnedSocialPolicy")
    return partial(kind,path,memory=memory,relationships=relationships)


def scenario(version):
    if type(version) is not int or version not in (1,2):raise ValueError("Unsupported scenario version")
    return importlib.import_module("training.social_arena"+("_v2" if version==2 else "")).Arena
