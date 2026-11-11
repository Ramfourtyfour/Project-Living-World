# Policy and action interface

`training.native.World` owns one in-process C++ world. `observe(actor_id)` returns
an immutable JSON-shaped local view. `step({actor_id: Action(...)})` advances one
tick. One dictionary entry per actor prevents accidental duplicate input; the
native batch API separately rejects duplicate actors. No entry means wait.

`ParallelWorldEnv.reset(seed=...)` returns `(observations, infos)` keyed by stable
integer actor ID. `step(actions)` returns observations, rewards, terminations,
truncations and infos, with keys for actors active at the beginning of that step.
The next `agents` excludes terminated and truncated actors. `possible_agents`
retains the original roster. This follows the **shape** of the
[PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/); it is not a
PettingZoo subclass and does not advertise spaces or package compatibility.

An environment action can be an ordinary `Action` or one of eight integer intents:
forage food, fetch water, eat, drink, rest, explore, gather wood, wait. Intent
execution chooses a waypoint using only visible or remembered resource locations.
The native authority still checks reach and stock; an intent grants no resources.
These are one-tick proposals rather than automatic completed multi-tick projects.

The learned policy consumes 22 own/local features; it gets no seed, actor ID,
group ID or debug state. A mask prevents unavailable consumption and gathering
when inventory is full. It does not use an oracle to locate distant nodes.
Weights are shared, but GRU hidden vectors are indexed by individual identity.
The training critic sees the same features in this pilot. No centralized critic
or inter-agent message encoder has been implemented yet.

See `simulation-contracts.md` for schemas, capacities, fairness limitations and
private data. Social actions exist at the native interface but are not in the
pilot's eight-intent action space. Social training is therefore explicitly pending.
