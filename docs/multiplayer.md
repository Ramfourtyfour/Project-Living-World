# Authoritative co-op design (not implemented)

The native engine is already the sole state authority. There is currently no
network listener, authentication/session protocol, native game client, encryption,
rollback/reconciliation or co-op gameplay. The desktop inspector is a trusted local
diagnostic process and its oracle is unsuitable for a remote client.

Future clients send actor-owned intents plus sequence IDs. The host binds each
session to a stable individual, validates ownership before queuing actions and
sends only `ObservationV1`-compatible visible state. Reconnect resumes the same
identity and request high-water mark. It never grants a free person or duplicate
inventory. Human world-entry mode is fixed at world creation, following the plan.

Implement a bounded message protocol after the world contracts stabilize. Test
spoofed actor IDs, duplicate/reordered input, stale acceptance, disconnect during
transfer, visibility privacy and two real native clients under injected latency.
Current in-process checks do not satisfy that gate. No network security claims are
made for the C ABI.
