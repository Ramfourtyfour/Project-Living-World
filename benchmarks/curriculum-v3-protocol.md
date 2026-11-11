# Predeclared third curriculum protocol

Development diagnosis uses seeds 9000..9003, separate from policy optimization.
The first 8 held-out worlds have now been inspected repeatedly; call them
**validation**, not an untouched final test. The sealed test partition for this
increment is 20000..20015 and must be run only after selecting the candidate.

Hypotheses from the diagnosis: the old learned policy sometimes dies with food or
water in inventory, suffers fatigue damage absent in the heuristic, and discovers
few alternative supplies. Food waste is actually lower than the heuristic, so
waste alone does not explain the failure. Improve state encoding, demonstration
coverage and training horizon, and test the resulting combined intervention.

- Keep the C++ world's normal/scarce configurations and both baselines unchanged.
- Add a versioned 38-feature, 64-unit recurrent policy. Extra features use only
  visible/remembered stock, visible crowding, remembered age, own item presence,
  consumption waste and urgent needs. Old artifacts retain their old encoder.
- Warm start: seed 217, 16 episodes, 32 people, 256 ticks, alternating normal/scarce
  demonstration worlds, 64 balanced imitation epochs. Exclude dead/padded actors
  from cloning losses. Save and retain this checkpoint separately.
- PPO: seed 217, 96 updates, horizon 512, population 32, scarce with every fourth
  episode normal. Training seeds 317..412. Use reward version 2: unchanged v1 plus
  a -0.02 cost per wasted unit of food/water's potential need reduction. No hidden
  information and no changes to simulation feasibility or resource generation.
- Optimizer: actor/encoder/GRU LR 0.00003, critic LR 0.001, value coefficient 0.05,
  entropy coefficient 0.005, 4 epochs, clip 0.2, gamma 0.99, GAE lambda 0.95.
- Validate the final warm-start and PPO policies on seeds 10000..10007, both normal
  and scarce, 32 people, 512 ticks. Use the final PPO candidate for sealed testing
  only if it passes the original declared comparison on validation. Do not choose
  a best seed or best intermediate epoch using test results.
- Gate: no normal-survival regression versus the heuristic; strictly greater
  scarcity survival and greater scarcity exploration. Require the comparison on
  both validation and the sealed seeds, conservation throughout, and inspectable
  action/observation recordings. If it fails, retain and report the failure.

This is a joint intervention, not an ablation proving which individual change
caused a gain. Social behavior, construction and emergent society remain unproven.

## Deployment and causal audit, fixed before inspecting sealed scores

Export the selected final weights unchanged to the declared C++ GRU runtime. First
require numerical conformance to the PyTorch reference on development seeds 9010
and 9011 (zero intent disagreement, max logit/hidden absolute error <= 0.0001).
Run the sealed comparison in that deployment runtime without site packages.
Record native/source policy hashes so deployment cannot silently change weights.

For causal exploration evidence, record **the first sealed seed, 20000**, for the
heuristic and selected native policy in scarce mode over 512 ticks. Do not select
an impressive seed after seeing results. Replay both private journals and count
accepted food/water extraction from sources outside each individual's initial
sight. Require actual new-source use, rather than treating grid coverage alone as
proof of useful exploration. Retain the full logs and aggregate/individual reports.
