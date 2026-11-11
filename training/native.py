"""Typed in-process C ABI. Never feed debug_state() to a deployed policy."""
import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
KINDS = {name: i for i, name in enumerate(("wait", "move", "inspect", "gather", "consume", "offer", "accept", "trade", "rest", "depart", "invite", "decline",
                                         "found_group","rename_group","request","answer_request","deposit","withdraw","place_cache","report_request"))}
RESOURCES = ("food", "water", "wood", "stone", "fiber")

class CAction(C.Structure):
    _fields_ = [("actor", C.c_uint64), ("request", C.c_uint64), ("kind", C.c_int32), ("target", C.c_uint64),
                *[(key, C.c_int32) for key in ("x", "y", "resource", "quantity", "ask_resource", "ask_quantity")]]

class CActionV2(C.Structure):
    _fields_=[("base",CAction),("text",C.c_char*64)]

@dataclass(frozen=True)
class Action:
    kind: str = "wait"
    target: int = 0
    x: int = 0
    y: int = 0
    resource: int = 0
    quantity: int = 1
    ask_resource: int = 0
    ask_quantity: int = 0
    text: str = ""

_lib = None

def library():
    global _lib
    if _lib is not None:
        return _lib
    filename = "livingworld.dll" if sys.platform == "win32" else "liblivingworld.dylib" if sys.platform == "darwin" else "liblivingworld.so"
    candidates = [Path(os.environ["LIVINGWORLD_LIBRARY"])] if "LIVINGWORLD_LIBRARY" in os.environ else [ROOT / "build" / filename, ROOT / "build" / "Release" / filename, ROOT / "build" / "Debug" / filename]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError("Native library missing. Run: python tools/build.py")
    lib = C.CDLL(str(path))
    lib.lw_create.argtypes = [C.c_uint64, C.c_int32]; lib.lw_create.restype = C.c_void_p
    lib.lw_create_scenario.argtypes = [C.c_uint64, C.c_int32, C.c_int32]; lib.lw_create_scenario.restype = C.c_void_p
    lib.lw_load.argtypes = [C.c_char_p]; lib.lw_load.restype = C.c_void_p
    lib.lw_destroy.argtypes = [C.c_void_p]; lib.lw_destroy.restype = None
    lib.lw_step.argtypes = [C.c_void_p, C.POINTER(CAction), C.c_int32]; lib.lw_step.restype = C.c_int32
    for name in ("lw_debug", "lw_save", "lw_hash", "lw_results"):
        fn = getattr(lib, name); fn.argtypes = [C.c_void_p]; fn.restype = C.c_char_p
    lib.lw_observe.argtypes = [C.c_void_p, C.c_uint64]; lib.lw_observe.restype = C.c_char_p
    lib.lw_error.argtypes = []; lib.lw_error.restype = C.c_char_p
    lib.lw_abi_version.argtypes = []; lib.lw_abi_version.restype = C.c_int32
    lib.lw_action_size.argtypes = []; lib.lw_action_size.restype = C.c_int32
    if lib.lw_abi_version() != 1 or lib.lw_action_size() != C.sizeof(CAction):
        raise RuntimeError("Native ABI mismatch; rebuild the library")
    if hasattr(lib,"lw_step_v2"):
        lib.lw_step_v2.argtypes=[C.c_void_p,C.POINTER(CActionV2),C.c_int32];lib.lw_step_v2.restype=C.c_int32
        lib.lw_create_rules.argtypes=[C.c_uint64,C.c_int32,C.c_int32,C.c_int32];lib.lw_create_rules.restype=C.c_void_p
        for name in ("lw_rules_version","lw_upgrade_rules"):
            fn=getattr(lib,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_int32
        lib.lw_action_v2_size.argtypes=[];lib.lw_action_v2_size.restype=C.c_int32
        if lib.lw_action_v2_size()!=C.sizeof(CActionV2):raise RuntimeError("Named action ABI mismatch")
    _lib = lib
    return lib

def native_fingerprint():
    """Exact loaded native binary identity, for reproducible checkpoints/experiments."""
    return hashlib.sha256(Path(library()._name).read_bytes()).hexdigest()

class World:
    def __init__(self, seed=1, population=32, *, snapshot=None, scenario=0,rules_version=2):
        if snapshot is not None and (not isinstance(snapshot,str) or "\0" in snapshot):
            raise ValueError("Snapshot must be a text record without NUL bytes")
        self.lib = library()
        self.handle=None
        if snapshot is not None:self.handle=self.lib.lw_load(snapshot.encode("utf-8"))
        elif hasattr(self.lib,"lw_create_rules"):self.handle=self.lib.lw_create_rules(seed,population,scenario,rules_version)
        elif rules_version==1:self.handle=self.lib.lw_create_scenario(seed,population,scenario)
        else:raise RuntimeError("Modern rules require rebuilding the simulation library; legacy replay may use rules_version=1")
        if not self.handle:
            raise ValueError(self.lib.lw_error().decode())
        # Registry access is confined to bootstrap. Policies get only observe().
        self.ids = tuple(p["id"] for p in self.debug_state()["people"])

    def close(self):
        if getattr(self, "handle", None):
            self.lib.lw_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def _text(self, name, *args):
        if not self.handle:
            raise RuntimeError("World is closed")
        value = getattr(self.lib, name)(self.handle, *args)
        if value is None:
            raise ValueError(self.lib.lw_error().decode())
        return value.decode("utf-8")

    def observe(self, actor):
        return json.loads(self._text("lw_observe", actor))

    def observations(self):
        return {i: self.observe(i) for i in self.ids}

    def debug_state(self):
        return json.loads(self._text("lw_debug"))

    def snapshot(self):
        return self._text("lw_save")

    def state_hash(self):
        return self._text("lw_hash")

    @property
    def rules_version(self):
        if not self.handle:raise RuntimeError("World is closed")
        return self.lib.lw_rules_version(self.handle) if hasattr(self.lib,"lw_rules_version") else 1

    def upgrade_rules(self):
        if not self.handle:raise RuntimeError("World is closed")
        if not hasattr(self.lib,"lw_upgrade_rules"):raise RuntimeError("Rebuild the simulation library before migrating")
        if self.lib.lw_upgrade_rules(self.handle):raise ValueError(self.lib.lw_error().decode())

    def step(self, actions, *, requests=None):
        """One intent per actor; unknown actor IDs are rejected by the native authority."""
        if not self.handle:
            raise RuntimeError("World is closed")
        if len(actions) > 256:
            raise ValueError("batch limit is 256")
        batch = []
        for actor, action in actions.items():
            if not isinstance(action, Action):
                raise TypeError("actions must contain Action values")
            request = (requests or {}).get(actor)
            if request is None:
                request = self.observe(actor)["self"]["last_request"] + 1
            if not isinstance(action.text,str) or "\0" in action.text:raise ValueError("Action label must be text without NUL")
            label=action.text.encode("utf-8")
            if len(label)>63:raise ValueError("Action label exceeds 63 bytes")
            base=CAction(actor, request, KINDS.get(action.kind, -1), action.target, action.x, action.y, action.resource, action.quantity, action.ask_resource, action.ask_quantity)
            if hasattr(self.lib,"lw_step_v2"):batch.append(CActionV2(base,label))
            else:
                if label:raise RuntimeError("Legacy action ABI does not support labels")
                batch.append(base)
        modern=hasattr(self.lib,"lw_step_v2")
        array = ((CActionV2 if modern else CAction) * len(batch))(*batch)
        if (self.lib.lw_step_v2 if modern else self.lib.lw_step)(self.handle, array, len(batch)):
            raise RuntimeError(self.lib.lw_error().decode())
        return json.loads(self._text("lw_results"))

    def save(self, path):
        """Atomic same-directory replacement: a failed write leaves the old save intact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(self.snapshot()); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
