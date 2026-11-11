# Rules 3: individual senses and dated knowledge

New Python worlds use rules 3. Existing rules-1 and rules-2 saves load in their
original modes, with the original action/result/hash behavior. The original C
creation/upgrade functions retain their published behavior; `lw_create_rules`
accepts 3 and `lw_upgrade_perception` explicitly upgrades older worlds.

## Sight

Each person has a saved facing vector and a 120-degree horizontal field of view.
`turn(x,y)` looks toward a world waypoint, costs one tick and moves no body.
Successful movement faces its destination. Zero-direction turns and invalid world
coordinates fail. Crouching halves movement from 2 m to 1 m per tick. These are
simulation actions; a future first-person renderer will interpolate presentation.

Sight additionally requires range, obstruction, target illumination and concealment.
The deterministic 1,200-tick cycle has full daylight through tick 400, sunset through
500, night at light 15 through 1,000, and dawn to 1,200. Unconcealed range is
`300 + 15 * ambient_light` centimetres: 18 m in full daylight and 5.25 m at night.
Generated vegetation patches reduce target visibility range by their density.
Crouching in cover adds 25 concealment points, capped at 90, with a 1 m minimum
recognition range. It does not reveal objects behind the viewer or through rocks.

Large rock landmarks become visible when their nearest surface is visible; the
center need not be within range. In this circular-geometry prototype, that reveals
the circle's navigation geometry. Known landmarks have stable IDs and saved, dated
memories. Navigation uses perceived or remembered geometry only. Unseen resources,
people, cache contents and new terrain are never filled from the global registry.

Public person sightings retain identity, last-seen position, then-observed
membership and alive/dead appearance. They contain no private health, needs,
inventory, controller state or remote updates. Sightings and landmark memories are
bounded to 32 each, refresh on ticks only, and expose age/source/confidence.
Previously known resource/cache memories retain their existing separate contracts.
An individual can remember a person who has since moved or changed agreement.

## Hearing

Accepted physical actions create sounds: ordinary footsteps (4.5 m), crouched
footsteps (1.5 m), gathering (9 m), cache construction (12 m), speech (7 m), and an
intentional `call` (40 m). A call transmits no arbitrary text or commands. Rock
occlusion halves audible range. Hearing does not require looking at the source.
Sounds use actual action positions and the listeners' final positions that tick.
Rejected actions, silent waiting and remote state changes create no such clues.

Each listener receives an **anonymous** clue with a local sequence number, kind,
time, the listener's own position at receipt, one of eight compass octants and a
distance interval (0–5 m, 5–15 m, 15–40 m). Octants have a 23-degree half-width.
Confidence is coarse by interval/muffling and decreases with age; it does not
encode precise hidden distance. Clues contain no source ID, exact source position,
group, task contents, resource, quantity or private inventory. Sounds never update
visual person sightings or resource memories. At most 16 clues survive, for up to
40 ticks. Dead individuals acquire no new sensory information.

Private structured proposals still communicate their declared information to their
intended recipient under the existing finite social interface. Bystanders hear only
a generic voice clue. A request never supplies remote movement or completion data;
the separate belief/report rules remain in force.

## Persistence and compatibility

`LIVING_WORLD_V3` appends checked facing/posture, landmark IDs/geometry memories,
person sightings, cover and anonymous hearing records to the prior world/social
records. Counts, identity references, direction normalization, locations, times,
confidence, categories and conservation are validated on load. Reads do not mutate
state. Complete checkpoints and durable action journals preserve the same senses.

`World.upgrade_rules(3)` preserves existing people, items, inventories, agreements,
requests and old knowledge. It assigns IDs to existing rock landmarks, establishes
current facing from the person's existing style, and observes only what that new
facing permits. It generates no new cover, caches or materials. Original saves are
untouched. Explicit complete-checkpoint migration supports old rules 1 and 2 only,
after original file hashes, world identity and matching controller state validate.
An unrecognized rules-3 binary change still requires a separately verified migration.

The frozen survival model was trained on rules 1. Its new-world intent execution
can turn/search from local information, but it has no learned hearing/light/facing
encoder. Historical scores remain historical. Current development measurements
must be reported separately; they cannot establish learned sensory adaptation.

This is horizontal, planar prototype perception. Vertical fields, local fire/torch
lighting, weather, material-specific acoustics, sound playback, 3D visual occlusion
and broader human playtests belong to the native-client/ecology milestones.
