# Internal notes: reflection loops in agent systems

Reflection (critic/self-review) loops let an agent iteratively improve its
own output, but they are one of the easiest patterns to build badly. Two
failure modes show up repeatedly in practice:

1. **No exit condition.** If the loop only exits when a score crosses a
   threshold, and the scoring model is noisy or miscalibrated, the loop can
   run far longer than intended, or never terminate at all.

2. **No diminishing-returns check.** Even with a max-iteration cap, a loop
   that always runs to the cap on every query is expensive by default, even
   when the second and third iterations changed almost nothing. Tracking
   score deltas between iterations and stopping early when improvement is
   marginal keeps typical-case cost far below worst-case cost.

A robust critic loop should combine three independent stop conditions: a
hard iteration ceiling, a quality threshold, and a diminishing-returns
check -- so that no single miscalibration (a bad score, a bad prompt) can
cause runaway cost.
