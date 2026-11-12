# Social curriculum v1: predeclared bounded research protocol

This extends the actual rules-3 world, not its engine-side decision rules. Every
learned person chooses independently from its own observation; shared weights do
not install a group controller. No live rule forces acceptance, departure, joining,
cooperation or a split. Earlier survival artifacts and reports remain frozen.

## Interface and model

A bounded parameterized action slate contains survival/exploration intents,
looking/approaching locally known people or communicated waypoints, finite social
proposals and replies, voluntary agreement actions and authorized cache transfers.
Only mechanical feasibility is masked. Unwise but affordable trade, invitation
acceptance and unilateral departure remain choices. No private other-person needs,
inventory, controller memory, unseen current position, world seed or numeric actor/
group identity enters the features. IDs resolve local handles only.

A standard shared GRU encodes local context; a shared small MLP scores each action
candidate against that context. Candidate ordering has no learned slot embeddings.
Hidden vectors remain separate per person and survive complete checkpoint/resume.
Live weights are fixed. Native inference must be checked against PyTorch before a
trained artifact is offered for dependency-free deployment.

Initialize the context encoder/GRU and eight survival logits from the frozen
GRU64 survival model, with new input columns initially zero. The shared candidate
scorer adds learned residual scores to those eight logits and scores social
proposals directly. This transfers learned survival skill; it does not override
the neural choice with a heuristic. Declare the source policy digest in artifacts.

## Training and evidence partitions

Development/training seeds are below 45,000, including previously inspected seeds.
Validation uses 50,000–50,015. The sealed comparison uses 60,000–60,031 and is not
opened until a single candidate/configuration is frozen after validation. Setup
steps, actual acquisitions, action/results, observations, controller assignments,
reward components and world hashes must be recorded. No mutation of inventories,
needs, resources, positions or group membership through a debug/snapshot back door
is allowed in the task generator: use ordinary world actions, including waiting.

Begin with C2 interactions: real gifts/reciprocal offers with changing personal
cost, help and refusal, and invitations whose value must be discovered through
actual storage access. Randomize personal reserves, need, local stock, partner
behavior and turn order. Use 4–16 individuals, varied partners and older snapshots;
do not bind policies/reward to birth-group labels. Retain normal survival episodes
to detect forgetting. Expand to free-running mixed worlds only after these work.

Warm start from declared individual heuristics if useful, followed by recurrent
PPO. Compare random choice, survival-only, social heuristic, the shared learned
policy, and a mixed-policy ensemble/league. Keep failed runs and their conditions.

## Reward and anti-exploit requirements

The primary reward is individual survival/health, useful consumption, exploration
and costs for failed or wasted actions. No reward for an offer, acceptance, gift,
promise, invitation, departure, joining, founding or group count. If a later mixed
reward adds social benefit, declare it separately and credit only verified useful
outcomes of actual items/tasks once; compare against the individual-only reward.
No pure birth-group total, free items, duplicated transfers or group-change bonus.

## Gate and limits

Report actual useful trade/help completion, costly/useless refusals, survival,
inventory/conservation, invalid actions, and voluntary membership changes with
their subsequent resource/need outcomes. On held-out tasks the frozen candidate
must make useful social choices in at least 75% of feasible beneficial cases,
refuse at least 75% of individually costly/useless cases, and change agreement
voluntarily when subsequent real access helps, while also declining some invitations.
Normal-world survival must remain at least 95% at 512 ticks. Every gate episode
must conserve and replay. Show results separately for unseen partner mixes and
ablations without recurrent memory/relationship features.

Passing a narrow task family alone is partial LW-008 evidence. Full LW-008 also
requires diverse-policy interaction and voluntary membership changes in whole
world scenarios. Learned weights, higher reward, a passing test fixture, one
social event or copied heuristic labels alone do not prove this gate or society.
