"""C++ recurrent inference with no PyTorch/NumPy imports in the live runtime."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys

from training.native import ROOT
from training.policies import features,features_v2,intent_mask,execute_intent

ARCHITECTURES={"gru48-v1":(22,48,1),"gru64-local-v2":(38,64,2)}
_library=None

def library():
    global _library
    if _library is not None:return _library
    name="livingworld_policy.dll" if sys.platform=="win32" else "liblivingworld_policy.dylib" if sys.platform=="darwin" else "liblivingworld_policy.so"
    candidates=[Path(os.environ["LIVINGWORLD_POLICY_LIBRARY"])] if "LIVINGWORLD_POLICY_LIBRARY" in os.environ else [ROOT/"build"/name,ROOT/"build"/"Release"/name,ROOT/"build"/"Debug"/name]
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:raise RuntimeError("Native inference library missing. Run python tools/build.py --inference-only")
    lib=C.CDLL(str(path))
    lib.lw_policy_abi.argtypes=[];lib.lw_policy_abi.restype=C.c_int32
    if lib.lw_policy_abi()!=1:raise ValueError("Native policy ABI mismatch")
    lib.lw_policy_load.argtypes=[C.c_char_p];lib.lw_policy_load.restype=C.c_void_p
    lib.lw_policy_destroy.argtypes=[C.c_void_p];lib.lw_policy_destroy.restype=None
    for name in ("lw_policy_inputs","lw_policy_hidden_size"):
        fn=getattr(lib,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_int32
    lib.lw_policy_decide.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_float),C.POINTER(C.c_uint8),C.c_int32,C.POINTER(C.c_int32),C.POINTER(C.c_float),C.POINTER(C.c_float)]
    lib.lw_policy_decide.restype=C.c_int32
    lib.lw_policy_get_hidden.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(C.c_float)];lib.lw_policy_get_hidden.restype=C.c_int32
    lib.lw_policy_set_hidden.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_float),C.c_int32];lib.lw_policy_set_hidden.restype=C.c_int32
    lib.lw_policy_error.argtypes=[];lib.lw_policy_error.restype=C.c_char_p
    _library=lib;return lib

class NativeLearnedPolicy:
    def __init__(self,path):
        path=Path(path);self.handle=None;self._ids=set()
        manifest_path=path.with_name(path.name+".json")
        if manifest_path.stat().st_size>65536 or path.stat().st_size>2*1024*1024:
            raise ValueError("Native artifact exceeds size limits")
        self.manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        m=self.manifest
        if m["version"]!=1 or m["format"]!="lwgru-v1" or m["architecture"] not in ARCHITECTURES:
            raise ValueError("Unsupported native policy artifact")
        self.inputs,self.hidden_size,self.feature_version=ARCHITECTURES[m["architecture"]]
        if m["feature_version"]!=self.feature_version or m["inputs"]!=self.inputs or m["hidden"]!=self.hidden_size or m["actions"]!=8:
            raise ValueError("Native policy contract mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=m["sha256"]:
            raise ValueError("Native model checksum mismatch")
        self.lib=library();self.runtime_sha256=hashlib.sha256(Path(self.lib._name).read_bytes()).hexdigest()
        self.handle=self.lib.lw_policy_load(str(path.resolve()).encode("utf-8"))
        if not self.handle:raise ValueError(self.lib.lw_policy_error().decode())
        if self.lib.lw_policy_inputs(self.handle)!=self.inputs or self.lib.lw_policy_hidden_size(self.handle)!=self.hidden_size:
            self.close();raise ValueError("Native model dimensions differ from manifest")
        self.name=f"native-{m['architecture']}-{m['sha256'][:12]}-{self.runtime_sha256[:12]}"

    def close(self):
        if getattr(self,"handle",None):self.lib.lw_policy_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def decide_features(self,ids,rows,masks):
        if not self.handle:raise RuntimeError("Native policy is closed")
        count=len(ids)
        if count>256 or len(rows)!=count or len(masks)!=count:
            raise ValueError("Invalid native batch size")
        if any(not isinstance(i,int) or not 0<i<2**64 for i in ids):raise ValueError("Invalid individual ID")
        if any(len(row)!=self.inputs for row in rows) or any(len(mask)!=8 or any(v not in (0,1,False,True) for v in mask) for mask in masks):
            raise ValueError("Invalid feature/mask shape")
        identity=(C.c_uint64*count)(*ids)
        inputs=(C.c_float*(count*self.inputs))(*(x for row in rows for x in row))
        mask=(C.c_uint8*(count*8))(*(v for row in masks for v in row))
        choices=(C.c_int32*count)();logits=(C.c_float*(count*8))();values=(C.c_float*count)()
        if self.lib.lw_policy_decide(self.handle,identity,inputs,mask,count,choices,logits,values):
            raise ValueError(self.lib.lw_policy_error().decode())
        self._ids.update(ids)
        return list(choices),[list(logits[j*8:(j+1)*8]) for j in range(count)],list(values)

    def intents(self,observations):
        ids=[i for i,o in observations.items() if o["self"]["alive"]]
        encoder={1:features,2:features_v2}[self.feature_version]
        choices,_,_=self.decide_features(ids,[encoder(observations[i]) for i in ids],[intent_mask(observations[i]) for i in ids])
        return dict(zip(ids,choices))

    def actions(self,observations):
        return {i:execute_intent(observations[i],intent) for i,intent in self.intents(observations).items()}

    def memory_snapshot(self):
        if not self.handle:raise RuntimeError("Native policy is closed")
        hidden={}
        for identity in sorted(self._ids):
            output=(C.c_float*self.hidden_size)()
            if self.lib.lw_policy_get_hidden(self.handle,identity,output):raise ValueError(self.lib.lw_policy_error().decode())
            hidden[str(identity)]=list(output)
        return {"policy_sha256":self.manifest["sha256"],"runtime_sha256":self.runtime_sha256,"hidden":hidden}

    def restore_memory(self,snapshot):
        if not self.handle:raise RuntimeError("Native policy is closed")
        if snapshot["policy_sha256"]!=self.manifest["sha256"] or snapshot["runtime_sha256"]!=self.runtime_sha256:
            raise ValueError("Native policy/runtime differs from saved memory")
        ids=[int(i) for i in snapshot["hidden"]]
        rows=list(snapshot["hidden"].values())
        if len(ids)>4096 or any(not 0<i<2**64 for i in ids) or any(len(row)!=self.hidden_size for row in rows):
            raise ValueError("Invalid native recurrent state")
        identity=(C.c_uint64*len(ids))(*ids);values=(C.c_float*(len(ids)*self.hidden_size))(*(v for row in rows for v in row))
        if self.lib.lw_policy_set_hidden(self.handle,identity,values,len(ids)):
            raise ValueError(self.lib.lw_policy_error().decode())
        self._ids=set(ids)
