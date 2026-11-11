# PROJECT LIVING WORLD
## Standalone Game — Master Design, AI Research & Implementation Plan

**Version:** 2.0 — 11 November 2026  
**Status:** Design specification / proposed architecture, not completed features  
**Repository:** New independent repository, suggested name `living-world`  
**Genre:** First-person cooperative survival RPG + persistent emergent-society simulation  
**Primary platforms:** Desktop first (Windows/Linux); others considered after native desktop stability  
**Source relationship:** A **new game**, not a fork, dependency, rewrite-in-place, or planned replacement of Shattered PD Multiplayer.

---

## 0. Executive vision

**One-sentence pitch:** Live in a procedurally generated fantasy wilderness where every intelligent NPC is an autonomous learning agent; people gather, build, cooperate, fall out, move, create new people through scarce magic, and change the world with or without you.

The player is a person in the ecosystem, not its predetermined hero. There are no permanent computer-controlled “teams.” Starting groups provide shared history and an initial home, but membership is contingent, trust is individual, and social structures can form, dissolve, merge, split, or turn hostile. NPCs have their own information and priorities, rather than receiving omniscient orders from a faction brain.

**The primary objective of the product** is a *fun survival RPG* whose worlds produce interesting histories because independent agents respond to real constraints. The primary technical novelty is individual multi-agent reinforcement learning (MARL) coupled to a persistent, authoritative world simulation.

### Immutable pillars
1. **One simulation, equal rules:** humans and intelligent NPCs consume the same resources and interact through the same validated world actions. Differences in controls and AI inference do not grant free items, hidden knowledge, or invulnerability.
2. **Individual autonomy:** every AI individual makes decisions from its own observations and memory; there is **no central policy ordering all members of a faction**. Agents may share model weights for compute efficiency without sharing decisions or personal memories.
3. **Groups are not destiny:** initial affiliation is familiarity, not an enforced allegiance. Members can refuse orders, leave, reconcile, switch communities, or form new ones.
4. **Real material consequences:** loot, tools, housing, settlements, construction, crafting, population creation, resurrection, and trade have real sources, storage, costs, and ownership.
5. **Escalation emerges, but challenge is designed:** regions present stable risk tiers; society and ecology add change. Avoid global HP inflation or automatic day-based enemy upgrades.
6. **A playable game first:** satisfying first-person movement, exploration, equipment, encounters, building, and co-op must exist without requiring a research breakthrough in emergent language or diplomacy.
7. **Evidence over aspiration:** learned cooperation, sophisticated societies, and performance are hypotheses to test, never features claimed merely because the policy is trained.

### Explicit non-goals for the first release
- Seamless planet-scale geography; millions of always-active agents; full natural-language cognition; continuously training a different huge neural network per creature; an MMORPG; realistic politics; complete ecosystem evolution; unrestricted user-generated text; porting Shattered PD code.
- No mandatory storyline, world-ending timer, forced faction membership, forced base construction, forced AI companions, or mandatory player conquest.

---

## 1. Player fantasy and core gameplay

### What the player does
The player appears in a newly generated region with modest supplies and a few discovered facts. They may travel alone, adventure with friends, create or recruit autonomous companions, establish a shelter, or participate in an existing community. They can choose short expeditions or long-distance travel. They are not automatically a faction leader: individual NPCs may listen to them, disagree, cooperate, or leave.

### Three nested loops

**Minute to minute (30 seconds–10 minutes):** move, scout, inspect signs, gather useful supplies, manage light and stamina, negotiate or evade a creature, survive an encounter, explore an interesting landmark, craft a field tool.

**Session (15–90 minutes):** prepare at camp, choose a need/opportunity, travel, discover a location or encounter, make consequential decisions, return or establish a new temporary base, upgrade gear, learn what nearby societies are doing.

**World scale (many sessions):** resources shift, settlements develop or collapse, individuals build histories and relationships, monster populations move, routes become safer or riskier, new locations and communities become strategically valuable.

### Player reasons to explore
1. **Resource gradients:** nearby renewable supplies sustain life; distant biomes and sites hold alternatives, rare materials, and specialized crafting ingredients.
2. **Equipment and mastery:** weapon families and armor develop by use; lost techniques and rare ability materials come from discovery, not simply grind.
3. **Knowledge:** maps, environmental clues, ancient structures, strange phenomena, recipes, and the history of local communities.
4. **Social opportunities:** trade partners, recruitable individuals, contested supply sites, changing alliances, rescue opportunities, and settlements in need.
5. **Strategic geography:** safe routes, fertile land, water, defensible locations, and access to crafting facilities.
6. **Ecological surprises:** unusual creature habitats, migrations, and temporary world events with observable causes.
7. **Player-directed goals:** establish a home, travel as a nomad, protect a friend, master a weapon, become a renowned explorer, or build a durable community.

Make early journeys useful without mandatory busywork. A player who never builds a permanent house can still progress using portable shelters, neutral trading places, and shared workstations.

### What success means
This is an open-ended world. No forced win screen. Optional long-form achievements include mapping a region, discovering a lost site, stabilizing a community, crafting signature equipment, or surviving a challenging expedition. Character death, world loss, and optional hardcore modes are separate rule settings, fixed when the world is created.

---

## 2. World structure and geography

### World topology
A persistent set of large **connected outdoor regions**, generated from a single reproducible seed. Regions contain continuous terrain; boundaries are streaming/transitions, not dungeon floors. Initial scope: one forest/grassland region large enough for several settlements and multiple resource niches. Later: mountains, wetland, ruins, highlands, corrupted lands, coastlines.

Each region contains:
- Terrain elevation, water, vegetation, weather and temperature profiles.
- Resource deposits: local renewable resources, finite mineral or magical deposits, and salvageable structures.
- Creature habitats with population/resource dependencies.
- Landmarks: ruins, abandoned farms, caves, shrines, old roads, natural shelters, defensible sites.
- Navigation affordances: walkable routes, cover, traversable slopes, passages, visibility, obstruction.
- Starting settlements and agent spawns generated once at world creation, with fair spatial access and viable survival supplies.

### Generation pipeline
1. Choose seed, world-rule preset, geography configuration, difficulty distribution, and initial population.
2. Generate height/water/biome fields and region links.
3. Place water and renewable resource capacity before population and settlements.
4. Place ruins, landmarks, unusual materials, and discoverable information.
5. Generate habitats and initial monster populations.
6. Place initial individuals/groups near enough survival resources and travel routes to avoid instant unwinnable starts.
7. Validate connectivity, safe initial arrival, obtainable starter survival resources, viable crafting chain, and reachable exploration incentives.
8. Persist only seed plus accepted generated state and subsequent modifications. No secret reroll each time a player revisits.

### Spatial persistence
Every constructed or destroyed object, deposited item, depleted resource, established trail, claimed storage location, corpse/revival record, and social event has a stable ID and survives saving/loading. Region unloading must not erase history. Distant simulation may use summarized updates but must preserve conservation of resources and identity.

### Time and weather
Shared world clock with day/night and gradual seasons only when beneficial. Weather affects travel, visibility, shelter, and resource production. Environment changes should be announced through readable in-world signs, not secret instantaneous penalties. Start with simple day/night and one weather variable, then expand after testing.

---

## 3. Autonomous individual model

### Fundamental entity schema
Each intelligent individual has a stable globally unique `individual_id` and:
- **Body/state:** position, movement, health, stamina, hunger, thirst, rest, injury, conditions, physical capabilities.
- **Possessions:** equipment IDs, inventory slots, stored resources, item reservations, claimed work output.
- **Knowledge:** directly observed locations, remembered landmarks, object properties, encounters, threats, known crafting recipes.
- **Personal memory:** recent event sequence, long-term episodic summaries, important individuals, beliefs and uncertainty.
- **Social state:** per-person trust/familiarity, affection/aversion if applicable as non-romantic social dispositions, promises, recent cooperation, shared history, alleged group affiliation, outstanding exchanges.
- **Policy state:** policy identifier/version, recurrent hidden state, individual traits or style embedding, decision cooldown and action commitment.
- **Progression:** learned game skills, equipment XP, known recipes, achievement/history journal.
- **Life cycle:** alive/downed/deceased/restored, death-event ID, resurrection eligibility, immutable provenance record.

These fields are **simulation mechanics**, not access to global truth. For example, an agent may *believe* another individual stole an item and be mistaken. The engine records actual ownership separately.

### Perception contract
Agents observe only what their senses and tools legitimately reveal. Sight requires current field of view, distance, obstruction, concealment, and illumination. Hearing provides source confidence and approximate location, not omniscient coordinates. Memory has age, confidence, and source provenance. Agents cannot query the world registry for hidden resources, unseen actors, unexplored structures, or private inventory. Even trained policies cannot receive them in live observations.

### Independent decisions
Every agent may choose actions asynchronously on its own decision cadence. The action is a request to the world: the simulation validates feasibility, range, materials, permissions, and timing. Agents can initiate, cancel where allowed, or complete tasks. They don't bypass collision or item ownership.

### Group membership is a relationship, not an AI command channel
The world may store a minimal factual membership agreement for gameplay logistics (shared storage permissions, settlement claims, invitations), but the faction ID is not an unchangeable identity or source of automatic obedience. Group-level banners and maps are summaries derived from individuals' agreements and actual occupancy.

---

## 4. Emergent social groups, alliances and conflict

### Starting conditions
At generation, place agents in several **starting communities**. They share shelter, familiarity, language tokens, and some memories. This establishes plausible initial social structure; it does not guarantee permanent cooperation or install a team controller.

### Relationship mechanics
Each pair of agents may independently track familiarity, trust, perceived reliability, remembered harm/help, exchanged resources, and expected future cooperation. Relationship updates arise from observable events; there is no magical worldwide reputation oracle. A trusts B does **not** imply B trusts A.

Possible behaviors, all optional and policy-mediated:
- Offer or request assistance or supplies; accept, refuse, counteroffer.
- Invite someone to share shelter/resources or join a group.
- Leave without permission, voluntarily change affiliation, move elsewhere.
- Establish a new camp with willing companions.
- Cooperate across group boundaries; trade with rivals.
- Disagree about resource use, reconcile, split, merge or compete.
- Exchange warnings or information, including inaccurate information if an explicit communication mechanism permits it.

### No scripted splits
Do **not** code `if trust < 0.3: split_faction()` as the principal social logic. Agents choose to leave or maintain agreements under their own policies. The game engine only enforces the physical/legal consequence of decisions (inventory transfers, shared storage revocation, building claims, travel). Starting groups can be renamed or cease to exist through record updates when nobody claims membership.

### Agency and commands
Humans are ordinary members in the same social system. A player can suggest a task ("guard this camp," "travel with me," "help gather food") via a finite intent/communication interface. NPCs can accept or reject; a request never directly overrides their policy. Avoid claiming unconstrained language understanding in v1; a structured dialogue/action menu is more reliable and trainable.

### Cooperation as an empirical outcome
An agent receiving positive shared outcomes may learn cooperation, but it might also learn selfish or avoidant behavior. The design supports these possibilities; it does not promise complex treaties or politics on day one. Test with counterfactual situations and unseen seeds. Introduce richer speech and commitments only after simpler exchanges work reliably.

---

## 5. Population magic, death and persistence

### Intelligent population creation
**Scroll of Awakening** creates exactly **one new intelligent individual** (not a copy of the caster) when crafted/used using a specified, finite recipe. The new person has a unique ID, initial capabilities, no invented personal past, and optional familiarity with the creator and nearby community. Their allegiance is not bound. The same rule applies to humans/AI factions, though human players do not automatically gain controllable extra characters.

Creation must consume real materials, craft time, and possibly workstation access. Balance cost through resource ecology, not an arbitrary hard faction-size cap. Artificial growth can still outpace consumption and cause scarcity; splitting groups redistributes individuals but does not alter total population.

### Incapacitation and revival
**Revival Potion** restores an eligible downed individual during a bounded in-world recovery window. It is a real consumable with validated ownership, range, once-only application, and interrupted-action semantics. Use player-friendly recovery rules in standard difficulty so a single unexpected encounter is not catastrophic.

### Death and rebirth
**Scroll of Rebirth** can restore an eligible deceased individual from a persistent **identity/death record**. Rebirth restores their saved identity and selected progression/memory according to explicit game rules; it must not create a duplicate active individual. Rebirth needs rarer resources and a world-accessible crafting prerequisite. The deceased identity cannot simultaneously exist in two locations, inventories or save snapshots. A reborn agent remains autonomous and may not share its rescuer's goals.

### Player roster and new human joins
Initial human players are members of the world’s starting population. For an existing persistent world, adding a *new* human-controlled character requires a defined world-entry rule. Candidate rule: a crafted Awakening vessel creates one new person which a joining player controls; existing human reconnects resume their original individual at no creation cost. Alternatively public worlds may reserve unclaimed starting slots. Choose one per world preset to prevent free extra starting labor and population exploits. Do not demand an AI companion for a solo player.

### Edge cases to test
- Two people attempt resurrection of one identity simultaneously → one succeeds, one fails safely.
- Disconnect while potion/crafting/transfer underway → no duplication or disappearing resources.
- A created member leaves immediately → creator owns no claim to control it.
- Deceased player reconnects → world rules handle spectator/rebirth, not silent free respawn.
- A created member cannot inherit a deceased person’s memory without explicit identity restoration.
- World rollback/replay must not duplicate awakening or rebirth transactions.

---

## 6. Survival, resources and economy

### Needs
Food, water, rest, environmental exposure, medical supplies, and shelter. Start with **three core needs** (food, water, rest) plus temperature/environmental threat as an expedition modifier. Each need affects capabilities smoothly, and the game signals impending trouble before severe consequences. Avoid micromanagement meters that require constant attention.

### Resources and conservation
All gathering draws from real nodes with capacities and regeneration rules. All crafting consumes actual ingredients and creates a specific item with a unique ID. Storage has physical location and access rules. Do not spawn supplies because an NPC policy chose to gather. An NPC can fail to locate or reach a needed resource.

Starter resource categories:
- Consumables: food, water, plant products.
- Construction: wood, stone, fibers, binding.
- Equipment: metals, hides, salvaged parts, basic magical components.
- Population magic: rarer essence/crystals/relic components acquired through exploration or sustained production.
- Knowledge: recipes, maps, inscriptions; a discovered recipe is information, not necessarily a consumable physical item.

### Trade and possession
Offer, inspect, accept, transfer; trades are atomic and validated. Shared storage requires permission based on group agreements; it is not magically communal. Possession and title can be disputed socially but the engine must always know current physical holder and transaction provenance. Item transfers survive reconnect.

### Population/ecology constraints
More individuals may produce more resources, but also consume food, land, transportation effort, gear and shelter. Logistics and local resource limits create natural pressures without scripted population ceilings. A single winning strategy (e.g. mass awakening) is a balance/test failure if it dominates across all environments.

---

## 7. Construction, settlements and land use

### Initial building system
Build foundations/camps, shelter, storage, campfire, workbench, water collector, garden plot, barricade, and alchemy station. Every construction project is a set of world actions with real site requirements, materials and progress. Individual agents can initiate and participate independently; no god-mode 'faction build' action instantly creates a village.

### Settlement definition
A settlement is a *place* with constructed infrastructure and persistent world records. It may contain residents with different social affiliations. An abandoned camp remains physically present until changed. Settlement names and group ownership arise from public agreement or de facto usage; do not tie lifetime to a permanent faction object.

### Property/access
Doors, containers, construction contributions, storage agreements and personal possessions use simple rules. A social promise can be broken; if a character takes an item without permission, the engine treats it as an actual action with observable consequences, not as an invisible deletion. Define limited v1 interaction scope before attempting elaborate law systems.

### Building AI
Provide discoverable building sites, construction plans, material reservations, hauling and placement actions. A policy can choose a goal; deterministic planners execute safe micro-actions. The result must remain a real, fallible world process. Validate connectivity, collision, blocking, and exploit-prone infinite construction loops.

---

## 8. Monsters, ecology and nonhuman threats

Monsters are autonomous inhabitants, not enemies spawned exclusively for human proximity. Some use ecological/rule-based behavior; selected intelligent monster families can use RL when useful. Predator territories, food availability, group size, seasonal movement, and environmental stress influence their actions. Different monsters pose different ecological and encounter challenges to human and AI communities alike.

Start with small fauna, neutral wildlife, a territorial monster, and one rare large creature. Ecological population changes need resource-dependent births/replacements; the **intelligent-person Awakening rule is separate** and remains magic-based. Avoid expensive per-frame neural inference for simple herd behavior unless evaluations justify it.

---

## 9. First-person survival gameplay and equipment

### Movement and interactions
Responsive first-person camera, continuous movement, physical bodies, grounded traversal, item inspection, interaction range, and readable contextual controls. Player input uses the **same authoritative action API** as NPCs, although the UI abstracts the commands differently.

### Encounters
Combat should be clear, responsive and grounded in position, timing and equipment differences. Different gear categories give different playstyles. Noncombat options—avoidance, hiding, retreat, negotiation with intelligent inhabitants—should remain viable. Human agents and AI agents obey the same stamina, item, hit validation, and recovery rules; do not make AI arbitrarily immune to player mechanics.

### Gear progression
Keep the *design idea* of classless equipment XP without copying Shattered PD code. Implement original gear families, randomized but bounded rolls, rarity with meaningful tradeoffs, earned XP on validated use, and limited ability choices. Avoid a bloated first-release content catalog. Start with a few mechanically distinct gear families and expand only after each is tested in player and AI hands.

### Crafting knowledge
Unlock progressively via discovery, experimentation within bounded recipes, trading for knowledge, and facilities. Scarce resources produce meaningful expeditions. Avoid pure vertical stat scaling; advancement should broaden strategic options.

### Other player-directed activities
Mapping, noncombat exploration, farming, improving a settlement, travelling with an autonomous companion, protecting a remote trade route. Do not require conquering everyone.

---

## 10. Difficulty and long-term world progression

**Difficulty is not simply time.** Combine the following:

1. **Geographic baseline:** more distant or hazardous regions contain more difficult environmental threats, creatures, and exploration challenges from world creation.
2. **Individual progress:** surviving agents accumulate gear, game skills, memories, discovered recipes, and infrastructure.
3. **Dynamic competition:** capable communities can monopolize sites, grow, weaken, move, cooperate or split; these changes create new risks and opportunities.
4. **Ecosystem conditions:** monster movements, weather, food supply and habitat disruption affect all inhabitants.
5. **Optional challenge destinations:** special ruins/creature habitats continue to offer goals even in otherwise peaceful worlds.

No global day-N enemy stat multiplier; no unconditional level-scaling to match the player. Players should be able to return to earlier regions and feel their progress. Prevent unfair collapse through readable warnings, multiple viable options, reasonable starter security, and a standard-mode recovery path.

### Progression metrics to tune
- Player time to first safe shelter, first meaningful discovery and first satisfying equipment upgrade.
- Encounter success/retreat rates by region and equipment level.
- Resource scarcity distribution across time and starting seeds.
- New player death/recovery frequency under standard rules.
- Number of strategic options available in dangerous regions.
- Frequency of changed regional conditions that are *observable* before becoming catastrophic.

---

## 11. AI training design — actual individual MARL

### Formal objective
Model the simulation as a partially observable stochastic multi-agent environment. Each individual `i` obtains a local observation `o_i`, maintains recurrent memory `h_i`, chooses a feasible action `a_i`, and receives experience/rewards determined by its own state and observable effects. The environment executes validated actions for all individuals and human players under one authoritative world clock.

### Observation design
**Structured features:** needs, immediate physical state, inventory/equipment, nearby observed entities, perceived threats, local resource nodes, remembered landmarks (with confidence/age), relationship summaries for known individuals, current task progress, received communications, weather, and local navigation info. Use entity lists or graph/attention encoding with padded masks rather than sending a giant map to the neural network. Do not expose hidden map truth, actual global group intents, secret inventories, or unobserved player positions.

**Memory:** recurrent state for short-term context plus an explicit episodic/semantic memory store. Keep source, time, confidence and capacity limits. Separate hidden memory from permanent game experience and equipment progression.

### Action space
Use a **parameterized, hierarchical action API**:
- Personal survival: eat/drink/rest/seek shelter.
- Exploration: choose waypoint, inspect visible object, follow known route, scout nearby region.
- Resources: harvest known node, haul, deposit, craft known recipe.
- Construction: select valid site, begin/continue project, repair.
- Social: request, offer, accept/refuse, invite, leave, announce intention, trade.
- Encounters: approach/withdraw, defend, use equipped ability or interact.
- Population: craft or use Awakening, Revival, Rebirth where eligibility permits.

A learned *intent* can be executed by standard pathfinding, collision and interaction code. No bespoke scripted faction strategy is inserted. Invalid actions get explicit feedback and modest cost, not a free action that can be spammed.

### Decision frequencies
Physics/combat state updates happen at the engine tick; social/resource decisions occur less often (e.g. roughly every 0.5–2 seconds in the prototype). Fast tactical responses can run more often for nearby agents. Batch neural inference across individuals; do not do independent Python RPC calls for each NPC each render frame.

### Learning method — start simple, not dogmatic
1. Build random and heuristic baseline agents first to prove world mechanics.
2. Train shared-parameter recurrent PPO/MAPPO-style policies using centralized training critics where useful and **strictly decentralized execution**. Shared weights do not imply a faction hive mind.
3. Introduce policy ensembles/league opponents and multiple seed distributions to avoid overfitting.
4. Compare cooperative, selfish, and mixed reward designs rather than asserting one solves society.
5. Add optional per-individual adapters only if gains exceed complexity and compute cost.
6. Consider off-policy alternatives only after stable evaluation; do not invent a novel RL architecture before benchmark evidence.

### Reward design
Candidate components (all provisional and normalized):
- Sustained survival and healthy resource state, with diminishing reward for idle survival.
- Attaining self-chosen, verified useful goals (finding needed resources, completing viable shelter, escaping legitimate threats).
- Long-horizon access to needed resources and future capabilities.
- Optional social benefit measured through *current interactions* and mutual outcomes, not original faction ID.
- Costs for unnecessary resource loss, preventable deaths, repeated invalid actions, and excessive time wasting.

Avoid a direct reward for "betray", "split faction", "kill others", "build 100 walls", or "create unlimited people". Such rewards would drive artificial and exploitative behavior. Avoid making the reward a pure 'group total' since that biases against leaving the birth faction. Use ablations and adversarial evaluation to detect reward hacking.

### Training curriculum
- **C0:** Single individual finds food/water and survives; no combat or other agents.
- **C1:** Scarce resources, navigation with imperfect information, inventory and shelter.
- **C2:** Two to eight individuals with voluntary trade, sharing and collaboration; no fixed team reward.
- **C3:** Multiple starting groups, mixed resource gradients, migration and social relationship updates.
- **C4:** Construction, crafting dependencies, limited population magic and death/rebirth.
- **C5:** Independent monsters, tactical encounters and more complex topography.
- **C6:** Human-like opponent distributions, diverse world seeds, partial disruptions, long-horizon persistence.

Advancing levels requires generalization gains, not training loss alone. Mix easier tasks with harder ones to avoid forgetting.

### Population-based training and diversity
Keep snapshots of older policies, seeded behavioral variations, and rival policies. Randomly mix them in training worlds. Evaluate on genuinely held-out seeds, unseen resource distributions, varied population sizes and opponent mixes. The aim is robust adaptive behavior, not one fragile self-play exploit. Faction labels are allowed as initial *context*, not neural policy ownership.

### Online learning (post-v1 research)
Default **fixed pretrained policy during live play**, with personal memories and skill changes. True updating of neural weights online is an optional research mode requiring versioning, rollback, reproducible logs, stability/security checks, compute budgets, and explicit per-world enablement. Do not market adaptive inference as online training.

---

## 12. Training/evaluation instrumentation

Every episode logs world seed, policy/version mapping, agent IDs, actions accepted/rejected, causal events, observed information, reward components, episodes, survival and resource metrics, social interactions, and save/replay hashes. Log **behavior** rather than only total reward.

### Primary scientific metrics
- Survival distribution and extinction rate on held-out worlds.
- Time-to-resource discovery; breadth of explored reachable areas.
- Sustainable resource balance and starvation frequency.
- Distinct independent cooperation events, mutual aid and trade completion.
- Voluntary membership changes, group formation/merger/split events where observable.
- Policy diversity and changes in social interaction networks over time.
- Capacity to recover from supply shocks and opponent changes.
- Generalization to unknown maps, population sizes and mixed policy opponents.
- Invalid-action rate, stuck-loop time, exploit attempts, CPU/GPU consumption.

### Baselines and required ablations
Compare random policy, basic survival heuristics, scripted baseline, shared recurrent RL, diverse RL league; with/without memory, with/without relationships, different reward compositions, and different visibility constraints. An increase in reward alone is not proof that social intelligence emerged.

### Demonstrations that actually matter
- Give two viable survival choices; demonstrate trained agent adapts to local need.
- Remove a resource source and observe alternative exploration instead of infinite waiting.
- Offer reciprocal trade with changing incentives; record acceptance/refusal based on personal state.
- Demonstrate a group split caused by independent agent choices, not a scripted threshold.
- Restart world from checkpoint; verify identity/memory/reputation persists, and valid actions remain deterministic.

---

## 13. Technical architecture (standalone)

### Recommended stack
- **Simulation:** C++20 native engine/library, no rendering dependency. Entity IDs, world state, deterministic actions and resource accounting reside here.
- **Training:** Python + PyTorch, with a batched native simulation binding. Consider PettingZoo-style Parallel API for experiments, then TorchRL or RLlib for training orchestration. Verify exact versions before locking dependencies.
- **Inference:** C++-hosted policy runtime using exported models (evaluate ONNX Runtime or libtorch based on performance/reliability), batched and versioned.
- **Client:** Godot 4 as a practical first-person 3D presentation option, communicating with simulation/server. Its rendering and toolchain can be swapped if an early visual prototype demonstrates need; do not force a custom graphics engine initially.
- **Networking:** authoritative dedicated server or locally hosted authoritative process. Multiplayer client sends user input/intents, not world states or rewards.
- **Persistence:** versioned save snapshots and append-only transaction/event logs for significant irreversible actions; content-addressed policy artifacts, migration utilities and rollback backups.
- **Build/CI:** CMake, platform builds, deterministic tests, sanitizers, headless performance benchmarks, reproducible training configs, smoke tests with two real clients.

### Major modules
```
living-world/
  README.md
  docs/
    vision.md
    game-design.md
    simulation-contracts.md
    agent-observation-action.md
    rl-training.md
    multiplayer.md
    milestones.md
    acceptance-tests.md
  sim/                 # C++ deterministic world and mechanics
    core/
    worldgen/
    entities/
    resources/
    construction/
    inventory/
    survival/
    combat/
    social/
    population/
    ecology/
    persistence/
  inference/           # Model loader, versioning, batch decision service
  training/            # Python env wrapper, PPO/MAPPO, league, evaluation
  client/              # Independent first-person presentation/UI
  server/              # Authoritative world host and network protocol
  tests/               # Unit, property, save/replay, adversarial, network
  benchmarks/          # Simulation throughput, inference cost and latency
  tools/               # Seed inspector, world replay, agent event explorer
```

### Authority boundaries
**Server/world:** simulation clock, collision, location, crafting, item ownership, death, population transactions, group/public agreements and consequence events.  
**Policy inference:** proposes actions based only on filtered observations.  
**Human clients:** propose actions and render authorized snapshots; cannot execute ownership or spawn operations locally.  
**Training:** has access to rewards and potentially global critic state, but no training-only oracle leaks into live policy observation.

### Data contracts (minimum)
`IndividualRecord`, `ItemRecord`, `RegionRecord`, `ResourceNode`, `BuildingRecord`, `RelationshipRecord`, `GroupAgreement`, `KnowledgeEvent`, `PopulationTransaction`, `ObservationV1`, `ActionRequestV1`, `ActionResultV1`, `PolicyArtifactV1`, `WorldSaveV1`.

Each contract documents identity, authority, validation, versioning, determinism expectations, serialization, nullability and abuse cases. Never use pointer addresses or object identity as save identifiers.

### Determinism policy
World generation and game mechanics seeded and replayable under the *same code/version, ordered inputs and policy outputs*. Floating-point and parallel scheduling can obstruct bitwise determinism across platforms, so either constrain them or define tolerant state-equivalence/replay checks. A saved game includes RNG state, policy version, pending actions and all identity-critical transactions.

---

## 14. Performance and scale

### Simulate at different levels of fidelity
- **Near humans:** full physics, line-of-sight, encounters, active crafting and close-range interaction.
- **Nearby but unseen:** lower-frequency navigation/agent decisions while preserving real line-of-sight rules and inventories.
- **Distant regions:** coarse-grained, event-based simulation of economic and travel outcomes, conserving resources and stable identities. If the coarse model cannot preserve consequences faithfully enough, pause selected events rather than fabricate results.

Crucially: an offscreen faction must not acquire knowledge or items from nowhere. Time acceleration must not provide free crafting/production, and unseen agent actions still require plausible travel and access.

### Initial measurable targets (engineering targets, not promises)
- Prototype: >= 32 individual agents per active world, with full action logs and reproducible rollouts.
- Early game: >= 64 active intelligent agents in region-scale stress tests with stable memory footprint.
- Performance: playable local co-op at 60 FPS target on an agreed reference desktop; p95 client frame time and server tick latency measured rather than assumed.
- Training: record environment steps per second, policy inference latency, memory use per agent and sample efficiency for each experiment.
- Scaling to hundreds/thousands of world individuals is a later milestone, gated by measured throughput and save reliability.

### Diagnostics
Built-in agent inspector (observation source, remembered beliefs, last decisions, action failures), relationship graph viewer, resource ledger, settlement timeline, deterministic world replay, training curves, policy comparison dashboards. These tools should be developed alongside the simulation, not after emergent behavior fails mysteriously.

---

## 15. Cooperative multiplayer and fairness

Human players share one authoritative persistent world. Their world rule preset is fixed at creation. Guests can connect/reconnect to their own stable characters; server verifies ownership of input identity. Clients only receive visible world state, remembered landmarks their character knows, and authorized social/settlement data. No invisible-agent tracking through network snapshots.

### Multiplayer behaviors
- Solo local host and two-player co-op for the first playable milestone.
- Save/rejoin with stable human character IDs and preserved pending actions.
- Item trades with atomic transfers and no duplicate items on reconnection.
- Player/NPC action fairness and consistent physical collision.
- Time passes for everyone in-session; pause rules explicitly defined for solo and multiplayer.
- Companions are autonomous, may decline requests, and continue operating according to world rules if their human acquaintance disconnects.
- New joining humans use the selected world entry rule (see Section 5), not silent free members.

### Anti-griefing and user experience
For private friend worlds, offer owner-selected permissions for destructive actions, shared storage and PvP, set at world creation. Avoid turning these into arbitrary bonuses for humans in simulation mode; they are **server governance presets** and should be transparent. Include accessible communication, subtitles/visual threat cues, and a way to view recent causal events without revealing unknown locations.

---

## 16. Art direction, atmosphere and narrative

**Visual identity:** stylized, grounded dark-fantasy wilderness with strong silhouettes, readable lighting, distinctive material surfaces, and a sense of place. Avoid an expensive photorealism target for v1. Make agent expressions and non-verbal signals legible so the player can understand fear, cooperation, suspicion and intent.

**World storytelling:** abandoned buildings, personal possessions, memorials, trade posts, community histories, roads, and visible construction progression. Procedural history emerges from recorded simulation events; don't fabricate a rich historical narrative that contradicts actual event logs.

**Audio:** footsteps, terrain, weather, environmental threat cues, creature calls, construction, and short social communication cues. Sound also feeds perception in the simulation, but gameplay hearing is distinct from speaker volume.

**UI:** survival needs, equipped gear, interact/offer/request menu, discovered map, settlement journal, personal history, relationship indicators when directly knowable, and a world event diary. Never expose precise hidden trust scores or future agent plans to players unless intentionally designed as an ability.

---

## 17. Concrete vertical slice (first genuinely playable experiment)

**Scope:** One finite but replayable forest region with grassland and ruins; 16–32 intelligent NPCs in four initial groups; 1–2 human players; a few nonhuman ecological threats; local food/water/resources; portable camps; one settlement type; minimal building; basic trade; a few original gear families; creation and revival items; save/rejoin.

**Minimum encounter:** Player leaves shelter, discovers a resource site, meets unknown individuals, can cooperate/trade/avoid, returns with tangible useful resources, and later discovers that the encountered group has changed its activities due to actual simulated events.

**Minimum RL demonstration:** On unseen seeds, learned policy beats a declared baseline at sustainable survival and demonstrates purposeful exploration under scarcity; social choices vary with conditions. No claim of 'emergent civilization' until measurable formation and persistence actually occur.

**Out of scope:** 1000+ agents; massive seamless world; free-form natural language; online neural weight updates; complicated politics; 120 creature families; 500+ gear abilities; cross-platform launch.

---

## 18. Detailed milestone roadmap and acceptance gates

All tasks belong in a new repo. Task IDs are **LW-001–LW-018**. Each is a separate testable increment, not a promise of a calendar delivery date.

### Foundation

**LW-001 — Vision, repo and independent licensing**
- Create new standalone repo, design contracts, CI, code style, supported platform matrix, license policy. No Shattered PD code or protected assets copied.
- **Gate:** clean C++ and Python builds, tests run in CI, docs clearly identify standalone product.

**LW-002 — Deterministic headless world**
- Clock, action/event queue, world seed, entity IDs, coordinate model, save/load/replay, transaction ledger.
- **Gate:** repeatable world-state hash under fixed version and ordered inputs; no duplicate IDs or resources after resume.

**LW-003 — Resource economy and survival**
- Node extraction/regrowth, inventories, food/water/rest, movement between sources, basic survival outcomes.
- **Gate:** conservation property tests, invalid gather/craft rejection, full survival loop from empty agent inventory.

**LW-004 — Individual perception and memory**
- Local sight/hearing, remembered observations, uncertainty and aging, observation API.
- **Gate:** no hidden entity/map/item leak in policy observations; perception and checkpoint tests.

**LW-005 — Social and group substrate**
- Individual relations, requests/offers, membership agreements, leave/join, shared storage access, no faction controller.
- **Gate:** two agents can disagree; group split/reunion driven by ordinary action requests; history persists.

### Training and economy

**LW-006 — Baseline agents and benchmark harness**
- Random/heuristic agents, PettingZoo-style wrapper, deterministic rollout, behavioral metrics and seed partitions.
- **Gate:** reproducible experiments and useful baseline reports.

**LW-007 — Independent MARL v1**
- Shared recurrent policy inference with individual observations/memory, PPO/MAPPO training, multi-seed evaluation.
- **Gate:** learned survival/exploration exceeds baseline on held-out worlds without reward exploit.

**LW-008 — Diverse social training**
- Trade, cooperation, individual trust, multi-policy self-play, breakable starting groups, league evaluation.
- **Gate:** measured useful social behavior and voluntary membership changes in test scenarios; no fake hardcoded faction triggers.

**LW-009 — Crafting and settlement construction**
- Building site validation, material logistics, shelter/storage/workstation/food production and self-initiated projects.
- **Gate:** independent agents complete a legitimate project and survive load/rejoin without inventory duplication.

**LW-010 — Population magic and life cycle**
- Awakening creation, downed/revival, deceased identity record and Rebirth, world-entry rules.
- **Gate:** one-use item transactions, exactly-one-identity resurrection, no policy or save exploits.

### Playable game

**LW-011 — Procedural regional world**
- Biomes, elevation/water, ruins, resource viability, generated routes and validation.
- **Gate:** batch of unseen generated seeds playable with reachable supplies and no sealed starting region.

**LW-012 — 3D exploration and survival client**
- First-person movement, world rendering, items, inventory, construction UI and journal.
- **Gate:** human completes gather/build/explore loop with same world action APIs as agents.

**LW-013 — Encounters, gear and ecology**
- Distinct equipment types, physical interaction, nonhuman creature behavior, AI-compatible equipment progression.
- **Gate:** fair actor/item mechanics; no AI-only shortcuts; observable monster interaction with human and AI groups.

**LW-014 — Authoritative co-op**
- Server protocol, host/join, ownership, visibility filtering, reconnect, client reconciliation.
- **Gate:** two native clients can gather, trade, explore, build and witness social/world consequences under injected lag.

### Scale and release

**LW-015 — Distant-world simulation and profiling**
- Simulation LOD, batched inference, region unload/reload, conservation-preserving aggregate events.
- **Gate:** far-region reactivation consistent with inventory/history ledger; measured performance improvement.

**LW-016 — World progression and balance**
- Hazard gradients, rare discovery chains, knowledge progression, flexible recovery and challenge balancing.
- **Gate:** playtests demonstrate meaningful exploration at early/mid/later sessions, not only starvation chores.

**LW-017 — Long-run society/economy evaluation**
- Hundreds of seeds at varying populations and opponent mixtures; anti-collapse and reward-hacking analysis.
- **Gate:** report distributions, not cherry-picked impressive runs; identify failure modes and policy regressions.

**LW-018 — Release candidate**
- Asset/license audit, platform packages, tutorials, options/accessibility, save migration, crash/latency soak tests.
- **Gate:** full native playthroughs, save/resume, reproducible distribution, clear known limitations.

### Critical dependency order
`LW-001 → 002 → 003 → 004 → 006 → 007 → 008`, with `005` before social training and `009/010` before comprehensive survival society trials. The 3D client begins after action/world contracts stabilize; multiplayer after the authoritative world can persist and replay. Ecology and performance iteration are continuous, but v1 proof must precede ambitious scaling.

---

## 19. Risk register and design safeguards

| Risk | Why it matters | Mitigation |
|---|---|---|
| Learned agents fail to cooperate or explore | Deep training does not guarantee society | Benchmarks, curriculum, clear action affordances, alternate policies |
| Reward hacking | Agents maximize a number rather than play plausibly | Component metrics, constrained actions, adversarial seeds and baseline comparisons |
| One policy dominates everything | Same outcomes each seed | Diverse policies, league self-play, economic realism, generalization tests |
| Population grows too rapidly | Scarce resources or exponential economy | Material-conserving creation, upkeep, supply chains, resource scarcity |
| Groups split, but simulation population still grows | Splitting does not remove members | Track global population and resource accounting separately |
| Agents appear omniscient | Breaks fairness and stealth | Strict observation filters, audit trajectories, packet privacy |
| Every agent has an expensive unique model | Training and real-time inference costs explode | Shared backbone + individual memory, batched inference, later adapters |
| World streaming fabricates resources/history | Trust and save integrity collapse | Causal ledger, conservative far-world simulation, reactivation tests |
| Open-ended gameplay becomes aimless | Players lack reasons to travel | Discovery/resource gradients, gear mastery, changing world opportunities |
| Player-hostile AI makes game unfair | Player loses agency | Readable warnings, safe region baselines, standard recovery, game rules |
| Social betrayal feels random | Players cannot understand decisions | Observed actions, memory/relationship feedback, comprehensible communications |
| Copying old game code creates license confusion | Standalone game must have clean provenance | Original code/art, asset manifests, license reviews |
| Stacked unfinished features create unstable platform | Hard to debug/fix | Small gated increments, tests before content expansion |

---

## 20. Quality gates and experimental claims

### Never claim complete from these alone
- A catalog entry exists, code compiles, reward rises, agent moves, test fixture passes, screenshot looks good, or a social event happened once.

### Require for each simulation feature
1. Documented world state and action contracts.
2. Unit/property tests, including malformed/duplicate commands.
3. Replay/save/load tests and identity/item conservation.
4. Behavioral test on varied seeds.
5. Performance profile and memory boundary.
6. At least one actual human-playable scenario once 3D client exists.
7. Co-op test if feature crosses the network.

### Suggested live acceptance scenario
Two human players establish a temporary camp, meet unfamiliar agents, trade supplies, travel to a resource site, return to find a group’s membership changed for a verifiable reason, craft an item through legitimate materials, witness independently acting agents, save, disconnect, reload, and continue with all identities, items, memories and structures intact.

### Scientific honesty
Report what is **implemented**, what is **observed in simulation**, what is **verified in human play**, and what remains **aspirational**. Keep policy runs, seeds and action logs inspectable so social claims can be reproduced.

---

## 21. First work package: exact next steps

**Goal:** a self-contained, replayable headless prototype—not an enormous engine.

**Deliverables:**
1. `World` with deterministic clock, 2D continuous coordinates (expand to 3D later), resources, body/entity IDs and save snapshots.
2. `Individual` with hunger/water/rest, inventory, local perception, memory, and one active intent.
3. `WorldAction` validation for move, inspect, gather, consume, offer, accept, trade, rest and depart group.
4. A scene generator that places 16–32 individuals in four initial groups with independently sampled environment/traits.
5. Logging and playback for events and personal observations.
6. Heuristic baseline policies that can stay alive in simple seeds.
7. Python environment binding with per-agent local observations and simultaneous step API.
8. First pretrained policy experiment, compared on held-out seeds to heuristic baseline.
9. Tests: resource conservation, no hidden observations, split/join not scripted, deterministic save/replay, no duplicate entity IDs.

**Stop/go criterion:** if a simple RL policy cannot outperform the heuristics in survival and useful exploration, do not proceed immediately to complicated settlement construction or 3D visuals. Improve environment design, rewards, observations and curriculum, then rerun evaluations.

---

## 22. Product identity summary

**Project Living World** is independent of Shattered PD Multiplayer. Its identity is not based on dungeon floors, traditional scripted enemy waves, or permanent starting factions.

Players are human members of a persistent world inhabited by individually autonomous agents. All intelligent people can survive, explore, gather, craft, trade, build, create new individuals with rare magic, resurrect known individuals, cooperate, migrate, leave groups, establish new communities, and change the world. Their decisions, constrained by the same world rules and the information they actually possess, determine the history of each simulation.

**Success looks like:** a fun first-person survival game where coming back to a familiar location feels different because real inhabitants have made consequential, independently chosen decisions while the player was elsewhere.
