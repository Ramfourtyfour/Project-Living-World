# Rules 2: agreements, personal relationships, requests and caches

New worlds use rules version 2. The native C ABI retains the original action struct
and functions for legacy clients, and adds `LWActionV2` with a bounded UTF-8 label.
Python `Action.text` supports group names; names are labels, not authority. A name
must be valid UTF-8, 1–40 bytes, without ASCII controls or leading/trailing spaces.
Wire-format failures reject the batch before time advances. Validly encoded but
infeasible actions return ordinary failures and cost their actor's tick.

## Individual agreements

Four starting agreements are generated, with neutral trust and familiar initial
members. An unaffiliated individual can found a named agreement. A member can
rename it, invite somebody, accept/decline an invitation or leave unilaterally.
Invitation acceptance is a separate action by the recipient. No founder controls
another person's policy. No trust threshold causes a scripted departure or split.
Empty agreements retain their history and cache claims; no items vanish when
members leave. Group IDs have a separate typed namespace from world entity IDs.

Names have revisions. Individuals retain only names learned initially, from
visible announcements/badges or from invitations. A remote rename is not a global
knowledge update, and observations contain no global roster/member positions.
All agreement and individual knowledge records survive save/resume.

Relations are directional, per person: familiarity, trust [-100,100], observed
help, broken promises and last interaction time. Received gifts and reciprocal
trades update the actual parties' experience. A witnessed donation may affect a
nearby member's opinion, never a remote member's. A declined request carries no
automatic trust penalty. These subjective states do not expose another person's
private inventory or future policy plan and do not drive engine-side group orders.
Ordinary human UI does not display others' precise private trust values.

## Requests are proposals

`request` has a recipient, intent and quantity. Intents are food (0), water (1),
visit (2), guard (3). Food/water requests ask for 1–4 units and retain the
sender's last communicated location. Visit/guard requests specify a waypoint;
their quantity is one. Requests and replies require sight and 12 m communication
range. A recipient explicitly answers yes/no using `answer_request` quantity 1/0.
Acceptance never moves the body, transfers resources or overrides the controller.

Statuses are pending (0), accepted (1), declined (2), fulfilled (3), expired (4).
The deadline is 60 ticks; terminal records remain for 80 additional ticks. Each
person may have eight active outgoing requests; the recent world pool is bounded
to 512 records. Historical causal records remain in the durable journal.

Food/water fulfillment requires actual accepted gift/trade transfers. Each
transferred unit contributes to at most one outstanding matching promise. A visit
requires reaching within 3 m of its waypoint; guard requires ten consecutive ticks
in that radius. These tasks can finish while the requester is away. There are no
threats yet, so guard is a presence contract, not demonstrated combat protection.

The performer sees actual progress. The requester sees a separate personal belief:
only witnessed progress or an in-person `report_request` can confirm completion.
Losing sight resets the requester's witnessed guard counter, without resetting the
performer's actual work. The requester's view contains no hidden completion time
or remote position updates. A report requires sight and 12 m communication range
and relays the true result; deceptive reports are not implemented.

An unverified accepted promise can expire in the requester's belief and reduce
personal trust even if the work succeeded offscreen. A later truthful report of
on-time completion corrects that mistaken broken-promise record and credits the
help once. Expiry discloses no reason for failure, changes no membership and issues
no commands. Actual state and both parties' knowledge survive save/resume. Request
records and replies are visible only to their two parties.

## Physical shared caches

Fresh worlds generate four empty camp caches. Each has a stable entity ID, group
claim, physical position, 120-unit capacity and embodied materials (3 wood,
2 fiber). That matter is included in the initial world budget. No carried food or
water is granted. Existing inventory ownership remains personal.

`place_cache` consumes 3 carried wood and 2 carried fiber in one work tick. It
requires membership, a site within 3 m, valid boundaries and space clear of
obstacles/caches/living bodies. Consumed ingredients become embodied matter rather
than disappearing. This is one minimal construction recipe, not the settlement
system in LW-009. Low caches block movement; the current sight model uses the
larger circular landmarks. Full 3D occlusion and inter-person collision are pending.

`deposit`/`withdraw` require current sight, 3 m reach, actual stock/capacity and the
current matching agreement. Both balances are validated before transfer. Leaving
revokes access immediately; it does not erase personal memories or seize the
leaver's carried inventory. Current cache contents appear only to an authorized
reachable observer. Remembered locations/contents have their own timestamps and
never refresh remotely. Cache memories are bounded to 32 places per individual.

Resource conservation is now:

`initial + regenerated = resource nodes + personal bags + cache stock + embodied materials + consumed`

The engine checks it every tick and on load. Starting infrastructure has an
explicit initial matter budget; construction and transfers create no free units.

## Save and replay versions

`WorldSaveV2` extends the validated v1 records with agreements, personal relations,
known name revisions, cache/place memories, requests and event-target redactions.
Events witnessed at the edge of sight redact target identities that the witness
could not see. Private rejections remain private. Versions and reference/bounds/
identity/material checks are enforced before a loaded world is returned.

Rules-1 worlds remain rules 1 on load and preserve their original save/action/hash
behavior. `World.upgrade_rules()` is explicit and idempotent. It preserves all
individuals, inventories and accounting, adds agreement metadata, spawns **no**
caches/materials and conservatively redacts unverifiable old event targets. The
original save is never rewritten by migration. A complete legacy checkpoint can
be upgraded in memory only after its original hash/IDs and matching controller
memory validate. A different neural-runtime identity still requires its own
explicit migration; a model name alone does not authorize it.

For a legacy journal, `--audit-compatible --verify-only` checks every recorded
result/hash in a read-only authority without running/restoring a controller.
`tools/migrate_journal.py` instead requires the matching controller, verifies the
old replay, saves a new complete checkpoint and can start a new journal. A migrated
audit handle cannot append to the old log. Migration/source hashes are recorded.

For a complete old neural bundle:

```text
python -S tools/migrate_checkpoint.py runs/native-checkpoint-128 runs/upgraded-checkpoint --native-policy inference/artifacts/curriculum-v3.lwgru --new-journal runs/upgraded.lwj
```

This verifies original file hashes, world identity and the matching controller,
preserves recurrent memory, and writes a separate bundle. Migration provenance
survives subsequent checkpoints and journal continuation. Existing destinations
are never overwritten.

## Evidence and remaining research

Integration tests cover independent founding/renaming, acceptance/refusal,
directional trust, deadline behavior, offscreen guard completion, private beliefs,
in-person reporting and late belief correction, no automatic movement,
hidden targets, private/stale cache stock, revoked access, actual cache materials,
named-wire validation and original-preserving migration. Sixteen surplus and
sixteen scarce-stock scenario seeds exercise an individual heuristic: it fulfilled
all surplus requests and declined all scarce requests; every log conserved and
replayed. These are mechanics/baseline results, **not learned social cooperation**.

`SocialHeuristicPolicy` makes each choice from one person's observation. The
pretrained GRU remains a survival/exploration controller with no social action
head. Next: learned social choices, diverse partners, counterfactual situations,
voluntary membership decisions, and separately declared held-out evaluation.
