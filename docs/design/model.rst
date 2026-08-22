The model
=========

*Where* a step goes and *which panels* it uses are different questions asked of the same music, so
both heads read one encoder rather than each learning its own view of the audio.

The encoder
-----------

Three ideas shape it.

**Difficulty modulates every layer.** A rating does not merely scale how many notes appear; it
decides which layer of the music to follow. Beginner charts track the kick, Challenge charts track
a drum fill or a vocal line. Feature-wise linear modulation applies a learnt scale and shift at
each block, so the rating can gate whole stems — which concatenating one embedding at the input
cannot do.

The conditioning vector is built from five embeddings: difficulty, rating, rating scale,
performance style, and the measured note rate. The last matters because a rating is a lossy
description of density: the classic ten-point scale saturates at the top, where charts labelled ten
run anywhere from 3.5 to 6.9 notes per second, and a keyboard chart runs roughly twice as dense as
a pad chart carrying the same number.

**Attention is local at the slot level and global at the measure level.** A song is a few thousand
slots, too many for full attention, but only a hundred or so measures. Structure is a measure-scale
phenomenon anyway: charters reuse a pattern when the music repeats, so a chorus should be able to
attend to the previous chorus. Slots see a 192-slot window; measures see everything.

**Style is conditioned on, not filtered out.** A pad chart is not a keyboard chart with its illegal
rows removed; it is composed differently, alternating feet and placing jumps on accents. Telling
the network which idiom it is writing in lets it learn each one, rather than having a decode-time
filter mangle patterns it never knew were disallowed.

The placement head
------------------

The placement head emits a **correction to a fixed metric prior**, not a probability. The empirical
odds of a step given its position in the bar are computed once from the corpus and added as a bias
the network cannot change, so rediscovering "notes fall on beats" earns it nothing. Whatever
accuracy it gains above that bias must come from the audio.

This is the property the first model lacked. It reached 0.986 overall AUC where a 48-entry lookup
table reached 0.978 — almost all of its apparent skill was metric position. The metric reported
during training is therefore the area under the curve computed *within* one metric position class,
where position carries no information at all.

At generation time only a quarter of the prior is kept. It is right about probability and wrong for
ranking: at full weight it puts quarter notes above everything else everywhere in the song, so
taking the highest scoring slots fills every quarter in the piece before it takes a single
sixteenth anywhere. Measured on one song at rating nine, the share of sixteenths and finer goes 1.3
per cent at full weight, 6 at a half and 13 at a quarter, against 13.2 per cent for real charts of
that rating.

The selection head
------------------

A row is four panel codes, so the space has ``7 ** 4`` members, but real charts use only a small,
heavily skewed subset. Each observed row is one vocabulary token, which lets the head learn jumps
and holds as single decisions rather than four independent ones — which is how charters think about
them.

The head is a causal transformer over the chosen steps, reading the encoder output at each step's
slot plus embeddings of the previous pattern, the gap since the previous step, and the position in
the bar.

The empty row is deliberately **not** in the vocabulary. It is the commonest pattern in any chart
and the one the selection head must never choose: where the rests go is placement's decision, and a
vocabulary entry for silence lets selection overrule it. Left in, it was taken at 28 per cent of
the slots placement had picked for a note, scattering holes through the chart.
