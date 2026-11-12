# Experimental individual social learning

The social action space is implemented. LW-008 remains open until useful learned
behavior generalizes across scenarios, partners and populations. The frozen
survival controller and v0.4 playable diagnostic package remain available.

Each individual now has a shared 64-unit GRU, its own hidden vector and a shared
candidate scorer. There are 25 action families: eight survival/exploration intents,
looking, approaching, gifts, quoted trades, requests and replies, invitation and
membership choices, storage transfers, reports, calls and crouching. At most 128
parameterized candidates are generated from that individual's local observation.
The scorer has no candidate-slot embeddings. Actor/group IDs resolve authorized
handles but are not numeric neural features. Other people's private stock, needs,
hidden memory and current unseen locations are absent.

Only mechanical feasibility is masked. Affordable harmful trades, refusing help,
leaving and accepting invitations remain choices. The neural scorer selects the
live action; no survival/social heuristic overrides that selection. Heuristics
are explicitly labelled demonstration or opponent policies. The transferred
survival head contributes learned logits to the same neural choice.

Version 1 uses 64 context and 52 candidate inputs. Version 2 appends twelve facts
about the observer's own needs, transaction cost, quoted own balances and remembered
storage access. It retains every v1 candidate and mask. Separate model identifiers,
manifests, native formats and runtime libraries prevent silent interface changes.
See the versioned protocols in `benchmarks/social-curriculum-v*-protocol.md`.

## Causal experiments and limitations

Six task families cover useful trade, costly trade, reciprocal help, useful and
costly membership invitations, and normal survival. Setup gathers, travels, waits,
consumes and builds through ordinary world commands. It never rewrites inventory,
need, membership or positions in snapshots. Every setup/control observation,
command, result and world hash is logged. Frozen-policy evaluation also reexecutes
neural decisions. An evaluation-only FIFO stock ledger attributes actual consumed
units to transfers; provenance and task success are never policy inputs or rewards.

Reward includes only individual survival, health/needs, useful rest, exploration,
waste and rejected actions. Giving, refusing, joining, leaving and group counts
receive no direct reward. Older learned snapshots, selfish survival baselines and
reciprocal baselines appear in the training mixture independently of birth groups.

The first candidate failed four inspected validation seeds: useful trade 100%,
costly-trade refusal 50%, reciprocal useful aid 25%, useful joining 75%, costly
invitation refusal 50%, normal survival 100%. The reciprocal task also had a defect:
the partner sometimes retained its requested resource, so gifts went unused. The
baseline exposed this failure. Version 2 creates mutual shortages through actual
upkeep and acquisition; its reciprocal baseline waits briefly and searches locally
before returning aid. The original failed run is preserved rather than relabelled.

On the corrected four-seed scenarios, that baseline succeeds in all six families.
The v1 learned policy still achieves only 50% costly-trade refusal, 50% reciprocal
aid, 75% useful joining and 50% costly-invitation refusal, with 100% normal survival.
These are diagnostic results on repeatedly inspected validation seeds, not sealed
generalization evidence. Current experiments use eight individuals; the broader
population and whole-world society gates remain open.

The second run collected 48 demonstrations, completed 24 imitation epochs and 96
recurrent PPO updates in 492.24 seconds. Final imitation accuracy was 93.49%; this
is training accuracy, not a behavior gate. Both checkpoints initially reached the
numeric thresholds on a four-seed diagnostic. Full 16-seed validation then gave:

| Measure | Declared social baseline | Imitation checkpoint | PPO checkpoint |
| --- | ---: | ---: | ---: |
| Useful trade success | 100% | 100% | 100% |
| Costly-trade refusal and survival | 100% | 68.75% | 87.50% |
| Both parties consume useful aid | 100% | 62.50% | 56.25% |
| Useful joining and storage consumption | 100% | 100% | 100% |
| Costly-invitation refusal and survival | 100% | 100% | 100% |
| Mean normal-world individual survival | 100% | 95.31% | 99.22% |

Both learned checkpoints **failed** the declared gate. All 96 episodes per
controller conserve and replay; learned decisions were reexecuted with frozen
weights. The last checkpoint improves costly-trade refusal and survival here,
while useful reciprocal aid declines. This is model selection on inspected
validation seeds. No sealed social result is claimed.

Recorded imitation failures include repeated looking/replies under critical hunger
or fatigue and some harmful accepted trades. The diagnosis reports actual actions
and preceding needs, not an optimal-action oracle. Preserve failures while improving
coverage of learner-visited states and longer individual survival. These results
do not establish that memory/relationships caused useful social behavior;
those contributions require controlled comparisons.

Training seeds stay below 45,000. Validation is 50,000–50,015; sealed evaluation is
60,000–60,031 and remains unopened. A diagnostic subset cannot pass the full gate.
The declared bounded thresholds require at least 75% success in **each** social
family, at least 95% normal survival, conservation and complete replay. Passing
those alone would still not complete LW-008 or establish emergent society.

## Reproduce

Training/export needs PyTorch/NumPy; native evaluation needs the standard library
and native world/social libraries. Preserve existing output directories.

```text
python tools/build.py --social-v2-only
python -m training.social_train_v2 --output runs/my-social-v2 --bc-episodes 48 --bc-epochs 24 --updates 96 --horizon 128
python -m inference.social_export_v2 runs/my-social-v2/policy.pt runs/my-social-v2/policy.lwslate
python tools/compare_social_inference.py runs/my-social-v2/policy.pt runs/my-social-v2/policy.lwslate --output runs/my-conformance.json
python -S tools/evaluate_social.py --native-policy runs/my-social-v2/policy.lwslate --partition validation --seeds 16 --output runs/my-validation.json
python -S tools/evaluate_social.py --baseline social --scenario-version 2 --seeds 16 --output runs/my-baseline.json
```

`--init-social` explicitly initializes v2 from a retained v1 learned artifact and
records its hash. Evaluation supports versioned older partners, memory/relationship
ablations and population settings; those comparisons must be measured before any
broader claim. Source bundles, training history and native/interface hashes bind
each experiment. The existing production survival artifact is never overwritten.

Both v2 checkpoints match PyTorch choices on 5,354 actual development decisions.
The PPO checkpoint's maximum logit error is 0.00001908 and hidden error 0.00000312.
The separate 8/32/64-person local latency report includes feature packing/encoding
and Python allocator peaks. Its 64-person end-to-end p95 was 145.35 ms while other
evaluation processes contended for CPU; it is not a client-frame/60-FPS result.
Hidden float payload is 256 bytes/person; total native allocator/map/DLL memory is
not measured by that figure. The frozen world and survival DLLs retain their exact
previous hashes. Source bundles, models, full reports and failed candidates remain
inspectable under `inference/artifacts/` and `benchmarks/`.
