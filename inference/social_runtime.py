"""Batched C++ social inference; standard library only during live execution."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys

from training.native import ROOT
from training.proposals import BASE_SIZE,CANDIDATE_SIZE,MAX_CANDIDATES,KINDS,context,proposals

_lib=None
INTERFACE=("training/proposals.py","training/policies.py","training/native.py")


def library():
    global _lib
    if _lib is not None:return _lib
    name="livingworld_social.dll" if sys.platform=="win32" else "liblivingworld_social.dylib" if sys.platform=="darwin" else "liblivingworld_social.so"
    candidates=[Path(os.environ["LIVINGWORLD_SOCIAL_LIBRARY"])] if "LIVINGWORLD_SOCIAL_LIBRARY" in os.environ else [ROOT/"build"/name,ROOT/"build/Release"/name,ROOT/"build/Debug"/name]
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:raise RuntimeError("Build the social runtime with python tools/build.py --social-inference-only")
    lib=C.CDLL(str(path));lib.lw_social_abi.argtypes=[];lib.lw_social_abi.restype=C.c_int32
    if lib.lw_social_abi()!=1:raise ValueError("Native social ABI mismatch")
    lib.lw_social_load.argtypes=[C.c_char_p];lib.lw_social_load.restype=C.c_void_p
    lib.lw_social_destroy.argtypes=[C.c_void_p];lib.lw_social_destroy.restype=None
    lib.lw_social_error.argtypes=[];lib.lw_social_error.restype=C.c_char_p
    lib.lw_social_decide.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_float),C.c_int32,C.POINTER(C.c_int32),C.POINTER(C.c_float),C.POINTER(C.c_uint8),C.c_int32,C.POINTER(C.c_int32),C.POINTER(C.c_float),C.POINTER(C.c_float)]
    lib.lw_social_decide.restype=C.c_int32
    lib.lw_social_get_hidden.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(C.c_float)];lib.lw_social_get_hidden.restype=C.c_int32
    lib.lw_social_set_hidden.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_float),C.c_int32];lib.lw_social_set_hidden.restype=C.c_int32
    _lib=lib;return lib


class NativeSocialPolicy:
    def __init__(self,path,*,relationships=True,memory=True):
        self.handle=None;self._ids=set();self.relationships=bool(relationships);self.use_memory=bool(memory)
        path=Path(path);sidecar=path.with_name(path.name+".json")
        try:
            if path.stat().st_size>1024*1024 or sidecar.stat().st_size>65536:raise ValueError("Oversized native social artifact")
            m=json.loads(sidecar.read_bytes());self.manifest=m
            if (m["version"]!=1 or m["format"]!="lwslate-v1" or m["architecture"]!="social-gru64-slate-v1" or m["observation_version"]!=3 or m["proposal_version"]!=1
                or m["base_inputs"]!=BASE_SIZE or m["candidate_inputs"]!=CANDIDATE_SIZE or m["hidden"]!=64 or m["max_candidates"]!=MAX_CANDIDATES or m["candidate_kinds"]!=list(KINDS)):
                raise ValueError("Unsupported native social contract")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=m["sha256"]:raise ValueError("Native social checksum mismatch")
            actual={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in INTERFACE}
            if m["interface_sources"]!=actual:raise ValueError("Social observation/proposal interface differs from export")
        except (KeyError,TypeError,AttributeError,UnicodeError,RecursionError) as error:raise ValueError("Malformed native social manifest") from error
        self.interface_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
        self.lib=library();self.runtime_sha256=hashlib.sha256(Path(self.lib._name).read_bytes()).hexdigest()
        self.name=f"native-social-{m['sha256']}-{self.runtime_sha256}-{self.interface_sha256}-r{int(self.relationships)}-m{int(self.use_memory)}"
        self.handle=self.lib.lw_social_load(str(path.resolve()).encode("utf-8"))
        if not self.handle:raise ValueError(self.lib.lw_social_error().decode())

    def close(self):
        if getattr(self,"handle",None):self.lib.lw_social_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def _set(self,rows):
        ids=list(rows);identity=(C.c_uint64*len(ids))(*ids);flat=(C.c_float*(len(ids)*64))(*(v for i in ids for v in rows[i]))
        if self.lib.lw_social_set_hidden(self.handle,identity,flat,len(ids)):raise ValueError(self.lib.lw_social_error().decode())
        self._ids=set(ids)

    def decide_features(self,ids,rows,candidates,masks):
        if not self.handle:raise RuntimeError("Social policy is closed")
        count=len(ids)
        if count>256 or any(type(i) is not int or not 0<i<2**64 for i in ids) or len(set(ids))!=count:raise ValueError("Invalid social identity batch")
        if len(rows)!=count or len(candidates)!=count or len(masks)!=count:raise ValueError("Invalid social batch shape")
        offsets=[0]
        for row,slate,mask in zip(rows,candidates,masks):
            if len(row)!=64 or not 1<=len(slate)<=MAX_CANDIDATES or len(mask)!=len(slate) or any(len(p)!=CANDIDATE_SIZE for p in slate):raise ValueError("Invalid proposal shape")
            if any(type(v) not in (int,bool) or v not in (0,1) for v in mask):raise ValueError("Invalid social action mask")
            offsets.append(offsets[-1]+len(slate))
        total=offsets[-1];identity=(C.c_uint64*count)(*ids);features=(C.c_float*(count*64))(*(v for row in rows for v in row))
        indices=(C.c_int32*(count+1))(*offsets);features_c=(C.c_float*(total*CANDIDATE_SIZE))(*(v for slate in candidates for p in slate for v in p))
        mask=(C.c_uint8*total)(*(v for m in masks for v in m));choices=(C.c_int32*count)();scores=(C.c_float*total)();values=(C.c_float*count)()
        previous=None
        if not self.use_memory:
            previous={int(i):v for i,v in self.memory_snapshot()["hidden"].items()};retained=dict(previous);retained.update({i:[0.0]*64 for i in ids});self._set(retained)
        try:
            if self.lib.lw_social_decide(self.handle,identity,features,count,indices,features_c,mask,total,choices,scores,values):raise ValueError(self.lib.lw_social_error().decode())
        except BaseException:
            if previous is not None:self._set(previous)
            raise
        self._ids.update(ids)
        return list(choices),[list(scores[offsets[j]:offsets[j+1]]) for j in range(count)],list(values)

    def decisions(self,observations):
        ids=[i for i,o in observations.items() if o["self"]["alive"]]
        if not ids:return {},{}
        slates=[proposals(observations[i],self.relationships) for i in ids]
        choices,_,_=self.decide_features(ids,[context(observations[i],self.relationships) for i in ids],
                                       [[p.features for p in slate] for slate in slates],[[p.enabled for p in slate] for slate in slates])
        return {i:slates[j][choices[j]].action for j,i in enumerate(ids)},{i:slates[j][choices[j]].kind for j,i in enumerate(ids)}

    def actions(self,observations):return self.decisions(observations)[0]

    def memory_snapshot(self):
        if not self.handle:raise RuntimeError("Social policy is closed")
        rows={}
        for i in sorted(self._ids):
            values=(C.c_float*64)()
            if self.lib.lw_social_get_hidden(self.handle,i,values):raise ValueError(self.lib.lw_social_error().decode())
            rows[str(i)]=list(values)
        return {"policy_sha256":self.manifest["sha256"],"runtime_sha256":self.runtime_sha256,"interface_sha256":self.interface_sha256,
                "relationships":self.relationships,"use_memory":self.use_memory,"hidden":rows}

    def restore_memory(self,snapshot):
        if (snapshot["policy_sha256"]!=self.manifest["sha256"] or snapshot["runtime_sha256"]!=self.runtime_sha256 or snapshot["interface_sha256"]!=self.interface_sha256
            or snapshot["relationships"]!=self.relationships or snapshot["use_memory"]!=self.use_memory):raise ValueError("Native social memory version differs")
        rows={}
        for raw,row in snapshot["hidden"].items():
            if not isinstance(raw,str) or not raw.isascii() or not raw.isdecimal() or str(int(raw))!=raw or not 0<int(raw)<2**64:raise ValueError("Invalid social memory identity")
            if not isinstance(row,list) or len(row)!=64:raise ValueError("Invalid social hidden shape")
            rows[int(raw)]=row
        if len(rows)>4096:raise ValueError("Social memory capacity exceeded")
        self._set(rows)
