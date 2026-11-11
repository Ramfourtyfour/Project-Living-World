# Predeclared second curriculum protocol

The normal-world heuristic has reached 100% survival. Preserve that distribution
and add a scarcity task rather than reporting reward alone. This protocol is
written before the scarcity evaluation is executed.

- Normal (scenario 0): unchanged generation, 32 people, horizon 512.
- Scarce (scenario 1): starting food/water node stock/capacity 12 rather than 40;
  food regrowth 2 per 20 ticks, water 2 per 12 ticks. Scattered resources remain
  unchanged. Total population 32, horizon 512. No automatic disaster or unseen
  source deletion; generation creates and saves the accepted resources once.
- Training: 32 individuals, scenario 1, seeds 117..148, 32 PPO updates maximum,
  starting from the separately evaluated balanced cloning checkpoint (seed 17).
- Evaluation: held-out seeds 10000..10007, all three policies on both scenarios.
- Record survival, lifetime, exploration, invalid actions, latency, conservation,
  configuration and hashes. Retain all original normal-world failures.
- Success requires no normal-survival regression, strictly higher scarcity
  survival than the heuristic and greater exploration on the scarcity set.
  This remains a small pilot gate, not proof of long-run emergent society.
- If the gate fails, keep the negative result and improve the curriculum or
  algorithm. Do not claim the full master plan is implemented.
