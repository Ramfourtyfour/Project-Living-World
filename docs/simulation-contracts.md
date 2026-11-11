# Simulation contracts: rules 1 and 2

This is an original standalone engine, unrelated to Shattered PD source or assets.
The C++ authority owns time, positions, inventories, membership and all transfers.
Python policies propose intents. The intent executor converts them into ordinary
validated actions. It can use only the current individual observation and memory.

New worlds use rules 2. Legacy rules-1 worlds retain their original save/action
behavior for replay and explicit migration. `social-contracts.md` specifies the
rules-2 extensions and their knowledge, storage and communication boundaries.

## Time, identity and coordinates

One step advances one world tick. Each individual gets at most one action. Missing
actions mean wait. Batches are resolved by ascending stable actor ID, independent
of input dictionary ordering. This is deterministic serialization, not simultaneous
physical resolution: lower IDs have a documented contention advantage. Duplicate
actors in a batch have *all* their requests rejected. A positive, monotonically
increasing request ID prevents a command being applied twice after reconnect.
Rejected unique requests also advance that actor's high-water mark.

Coordinates are integer centimetres on a 240 m square plane. Movement travels at
most 2 m per tick toward a waypoint. A 35 cm body cannot cross circular obstacles.
Individuals currently do not collide with one another. The floating-point segment
calculation means bitwise cross-platform replay is not promised. Same-build replay
and checkpoint resumption are checked. Generation RNG is explicit xorshift64*.

All people, nodes, offers, requests, caches and events use one monotonic uint64 ID
allocator. Group agreements occupy a separate typed integer namespace, starting
at 1..4 and growing through individual founding; 0 means no agreement. There
are no pointer-based save IDs. Dead individuals retain identity and inventory;
death does not mint a replacement. Creation/revival is a later milestone.

## Record and observation schemas

`IndividualRecord`: ID, position, public membership, life state, own health/needs,
five-slot resource inventory, sampled style, request high-water mark, memories and
directional familiarity. Rules 2 adds private directional trust/reliability,
remembered cache places and locally learned agreement labels. Population is
bounded to 64 for this prototype.

`ResourceNode`: ID, resource kind, position, current capacity, maximum and regrowth
period. Resource kinds are food, water, wood, stone, fiber. All quantities are
integers. No actor starts with carried inventory. The initial matter budget
includes natural nodes and, in rules 2, the materials embodied in four empty camp
caches. Regrowth is the only later material source, explicitly added to the ledger.
Scenario 1 is a declared scarcity curriculum: camp food/water capacity starts at
12, with food regrowth period 20 and water period 12. Normal scenario 0 uses 40,
periods 5 and 3. Resulting nodes and ledger, rather than a mutable difficulty flag,
are authoritative in the saved world. Scattered nodes follow the same rules in
both scenarios. There is no free runtime replenishment on scenario changes.

`ObservationV1`: own record; currently visible nodes and public body records;
remembered node sightings; own incoming/outgoing offers; known relationships;
visible obstacles; up to 32 witnessed events. Never another person's needs,
inventory, policy state, private offers or unseen position. Currently sight is
omnidirectional over 18 m with circle obstruction tests. FOV, light, concealment
and hearing are **not implemented**. A remembered location is not live truth.
Memories retain source, age, last observed amount and a decaying confidence proxy.
Memory refresh happens on world ticks, never on reads. Capacity is 32 sightings.

`ObservationV2` retains those fields and adds locally known agreement labels,
private requests, visible caches and 32 timestamped place memories. Current cache
contents require authorized reach. Remote names, contents and unseen event-target
IDs do not become global knowledge. The survival feature encoders project only
their declared old/local fields; social learning still needs its own action head.

`ActionRequestV1`: actor, request ID, kind, target ID, waypoint, resource/quantity,
requested resource/quantity. Public C ABI is in `sim/api.h`; Python `Action` is in
`training/native.py`. Unknown kinds and malformed quantities fail explicitly.
The additive named-action ABI `LWActionV2` supports the rules-2 operations without
changing the original struct layout. Names are bounded UTF-8, not executable text.

`ActionResultV1`: actor, request ID, success, reason. Failures do not transfer items
and cost world time. Gather, inspect and offer enforce current visibility; gather
and social exchange require 3 m proximity. Max gathering is 2 units per tick.
Total inventory capacity is 24 units. Consume accepts one food or water unit.
Food reduces hunger by 45, water reduces thirst by 55; rest reduces fatigue by 9
before the normal per-tick needs increment. At 100, needs damage health. There is
no health regeneration, crafting, exposure or shelter in v1. Rules 2 adds one
real-material cache construction recipe and validated physical cache transfers.

## Trade, offers and membership

Gifts and reciprocal trades create offers lasting 40 ticks. They do not reserve
inventory. Acceptance revalidates both people, reach, line of sight, expiry,
ownership and final inventory capacities before committing both balances. A spent
offer fails safely. Successful acceptance consumes the offer exactly once. Each
sender may retain at most eight outstanding offers. There is no private inventory
inspection API. Rules 1 records familiarity; rules 2 adds directional perceived
trust/help/reliability from actual parties' experience and witnessed events.

Invitations also require acceptance by the invited person. Depart is unilateral.
The engine never branches on a trust threshold or orders a group to split. Four
starting agreements have no faction controller. Baseline survival policies do not
choose social actions; voluntary membership tests exercise the mechanism, **not
learned social behavior**. Rules 2 supports individual founding/renaming, explicit
requests and shared storage; its social heuristic is a declared baseline.

## Event, persistence and policy boundaries

Events record ID, tick, actor, target, type, quantity and the set of witnesses at
emission. Rejections are private to their actor. The native recent-event ring holds
4096 events; it is not an unlimited world history. Full rollout JSONL captures every
step's actions/results/hash and personal observations. The durable single-writer
append-only action journal is implemented. Production compaction/backpressure and
the network service remain future work.

`WorldSaveV1` is a canonical, versioned text snapshot: RNG, clock, next ID, every
individual and possession, memory/familiarity, nodes, obstacles, pending offers,
recent events, last results and conservation ledger. It has an FNV-1a integrity
checksum (not cryptographic authentication). Loading checks structural limits,
identity uniqueness, references, bounds and conservation. Unknown versions fail;
no silent migration. Saving uses atomic same-directory replacement. No in-flight
multi-tick actions exist in v1. World snapshots exclude neural runtime memory;
`LearnedPolicy.memory_snapshot/restore_memory` persists it separately with the
policy SHA-256. `training/checkpoint.py` now publishes immutable bundles binding
world and controller files with an atomic manifest; `training/journal.py` durably
records and replays authoritative ticks and individual recurrent state. See
`durable-authority.md` for commit ordering, file leases and interrupted-write recovery.
`WorldSaveV2` appends agreements, caches, personal knowledge/relations, requests and
event redactions. Verified legacy migration retains people/materials and spawns
no caches. Compatible journal audits verify all old hashes and remain read-only.

`PolicyArtifactV1`: tensor weights plus JSON manifest: architecture, observation
version, intent names, SHA-256, method, training seeds/config/version/metrics.
Loading verifies the digest and uses PyTorch weights-only deserialization. Live
weights stay fixed; recurrent memory is individual and does not update weights.

`debug_state()` is an explicitly marked oracle for tests, diagnostics and metrics.
It must never be sent to an ordinary client or policy. This C ABI is an in-process
trusted boundary, not a secure network protocol. Native world objects are not
thread-safe. String return values are thread-local until the next string call.

## Accounting invariant

For each resource: `initial + regenerated = nodes + all inventories + embodied materials + consumed`.
Dead inventory counts. Each tick and save load checks this invariant. Offers are
proposals, not extra inventories. Cache ingredients move from bags into embodied
matter. No creation action bypasses conservation; population magic is pending.
