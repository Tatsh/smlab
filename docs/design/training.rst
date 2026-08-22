Training
========

The chart model
---------------

Examples are windows of 64 measures drawn from a cache of stem features. Entries are read from disk
per window rather than held in memory, because the cache runs to fourteen gigabytes, and it is
written **uncompressed** deliberately: reading one window out of a compressed archive costs 20 ms
because the whole feature array has to be inflated first, against 1.5 ms uncompressed. Passing
``mmap_mode`` does not rescue the compressed case, since numpy ignores it there rather than
reporting that it cannot comply.

Two details of the sampling matter more than they look.

**The rating scale is inferred from the pack.** It is not recorded in a simfile, so a pack whose
charts never exceed ten is taken to use the classic scale and one that reaches thirteen or more the
twenty-point scale. Without this the same number means two different difficulties and the
conditioning is ambiguous by construction. Packs that top out at eleven or twelve are dropped: too
high for one scale, too low to prove the other.

**Styles are sampled evenly.** The corpus is 95% charts danceable with two feet, so uniform
sampling would give keyboard and hand-chord charts about three per cent of the gradient updates and
their conditioning would go essentially unlearnt.

Every training window is shown under one of the pad's four reflections, chosen at random; held-out
windows are never mirrored, so validation figures stay comparable across runs.

The random generator is created per worker rather than in ``__init__``. A generator made up front
is copied into every worker process along with its state, so all of them draw the same window
starts in lockstep — with six workers the model sees a sixth of the window variety it should, and
the run cannot be reproduced from the seed alone.

The offset model
----------------

Labels are free. A song with a declared constant tempo and a declared offset tells us exactly where
its downbeats fall, so every excerpt of it is a labelled example and the label is whatever phase
that excerpt starts at. Folding a different excerpt, or from a different starting point, produces
another example of the same song at another phase, which is what makes 794 usable songs enough to
train on.

Only songs whose tempo is constant, whose offset is declared rather than guessed, and which carry
no stops are used. Anything else has a beat grid that either moves or was never authored.

Error is measured **around the bar**: bin 95 and bin 0 are neighbours, not 95 apart, and measuring
the long way round would make a near-perfect answer look like the worst possible one.

What is reported
----------------

The headline metric from the first chart model is deliberately not reported. An area under the
curve computed across all slots is dominated by the fact that notes fall on beats, which a
forty-eight entry lookup table already knows. What is reported instead is the same measure computed
*within* one metric position class, where position carries no information and any discrimination
must come from the audio.

A window where every quarter carries a step, or none does, says nothing about discrimination, so it
contributes nothing rather than being averaged in — and a run whose score comes out undefined must
not overwrite the checkpoint.
