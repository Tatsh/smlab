Timing
======

Everything downstream is indexed by beat, so the beat grid has to be recovered before any of it
means anything. For a track at a fixed tempo that grid is ``t_k = phase + k * 60 / bpm``: two
numbers, and the phase is exactly what ``#OFFSET`` encodes.

Tempo
-----

The tempo comes from **folding** the onset envelope onto a candidate beat period and taking the
mean onset strength at the beat phase. That picks the right *octave*: a grid at twice the true
tempo necessarily samples the weak off-beats as well, which drags its mean down, whereas a grid at
half the true tempo merely ties and is separated by a log-normal tempo prior.

Only a 25-second excerpt is folded, taken from the middle so intros and silence do not dominate.
Shorter is better here, which is counter-intuitive until the drift arithmetic is written down: a
candidate must stay phase-aligned across the whole span, and the drift it accumulates is the span
times the relative tempo error. A short span tolerates a coarser grid, so it resolves the right
neighbourhood more often. Measured across a training split, accuracy within one beat per minute
rises from 60% at whole-track to 84% at twenty-five seconds, with a clear interior maximum: 76% at
ten seconds, 72% at sixty.

An earlier version sharpened the answer with a Fourier peak inside one per cent of the fold's pick.
Over two held-out halves that stage fixed no song and broke fourteen, dropping accuracy from 98.6
to 84.3 per cent. With nothing left to gain it could only wander off a tempo that was already
right, so it was removed.

One tempo per song, unless you say otherwise
--------------------------------------------

Detecting tempo changes automatically was attempted and did not survive measurement. Of 3842 corpus
songs, 60 per cent declare one tempo and a further 17 per cent declare several within five per cent
of each other, which is an author correcting drift rather than the music changing.

Deciding windows jointly, by a Viterbi pass over the tempo grid with a penalty for changing, beats
grouping windows that agree: at a 5 per cent false-split rate it finds a change in 40 per cent of
songs that have one, against 10 per cent for grouping. It is still not enough. On the most
favourable population — two to four clean tempi, no gimmicks — it recovers every tempo and no
extras in 5.3 per cent of cases and returns a *partial* answer in 84.

A partial answer is the worst outcome available. A chart pinned to one wrong tempo drifts
predictably and ``--bpm`` fixes it; a chart whose grid is right for two minutes and wrong for one
cannot be corrected by any single number.

What failed there was the guessing, not the representation. :py:class:`~smlab.timing.TimingData` has
always been segment-based, and the writers have always emitted multi-segment ``#BPMS``, so a song
that drifts can be charted correctly the moment somebody says where it drifts.

:py:mod:`smlab.warp` measures that. It tracks the beat phase against a fixed reference tempo, and
where the music runs faster or slower the tracked phase slides; the slope of the slide is the tempo
difference and the sign says which way. ``smlab drift`` prints it per stretch along with the slip
each one accumulates, and ``--warp SECONDS:BPM`` on ``generate`` places a marker.

Choosing the segments
~~~~~~~~~~~~~~~~~~~~~

A tempo held over a stretch is a straight line in the tracked phase, so the fewest tempi that
describe a song are the fewest straight pieces the phase can be cut into with none of them straying
further than a tolerance. Every boundary is chosen at once, by dynamic programming over the
measurements, and ties on the number of pieces are settled by squared error.

Taking each piece as far as it will go left to right is the obvious alternative and it is wrong. A
piece runs past the change before the tolerance notices, the next piece then straddles the change,
and the fit reports a tempo belonging to neither side. On a click track stepping cleanly from 128 to
130 BPM, the greedy fit invents that bridge; choosing the boundaries jointly does not.

The tempi are then read off a single line that bends at those boundaries, not off each piece
separately. This is the part that matters most and the part that is easiest to get wrong. Pieces
fitted independently each choose their own starting height, so every boundary quietly carries a
jump in phase, and a tempo cannot jump. Writing those tempi into a file puts the grid out by the
sum of the jumps: measured on a song whose beat wanders by tens of milliseconds, the independent
fit accumulated 234 ms of error by the end, which is worse than writing no tempo change at all.
Bending one continuous line makes the fitted phase and the phase the chart will really have the
same object, and the same song then lands within 43 ms throughout.

Least squares is used for that line despite the tolerance being a worst-case quantity. Reweighting
towards the furthest-out measurements, to chase the smallest worst miss instead, was implemented
and measured: it made every case worse, because the weights collapse onto a handful of points. The
squared fit does lean about a tenth of a beat per minute into a phase step, which is the price of
being stable everywhere else.

What is left over after all of that is reported. A grid that still misses by more than the
tolerance is saying that the beat moves in ways no tempo describes, and no arrangement of warps
will recover it.

What a tempo segment must not be used for
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A segment can only say the beat spacing changed. That makes it the wrong instrument for a *step* in
the measured phase, and steps are common: where the beat appears to sit depends on what is playing,
so an arrangement change moves the measurement without moving the beat. Rendered audio at a
mathematically constant tempo demonstrates it. A click track with off-beat percussion entering
half-way through reports a spurious excursion of 0.8 BPM at exactly that moment under a naive fit,
and the same track without the entry reports one tempo to three decimal places.

Two rules keep those out. A boundary must move the grid by more than the tolerance across the
shorter of the two stretches it separates, so a boundary that exists only to absorb a step, whose
two sides differ by a few hundredths of a beat per minute, is dropped. Where the step is large
enough to survive that, the shape gives it away: the tempo departs and returns. Real tempo changes
do not undo themselves, so a tempo excursion that comes back to where it started is joined up. What
remains is deliberately conservative — a phase step is left uncorrected, because no tempo marker can
correct one.

A step joined up this way is reported rather than quietly absorbed, because it is a fact about the
audio worth knowing: the beat moved without changing speed, which is what an edit spliced into a
recording looks like, and the grid is unreliable around it however the warps are placed. ``smlab
drift`` and ``generate`` both say where those are, and both say that no warp will repair them.

Where the bends go is settled on stretches judged separately, but the tempi come off the single
bent line, so the two can disagree: a bend that looked worthwhile may end up separating tempi that
no longer differ enough to be worth a marker. Those are dropped and the line refitted, until every
remaining bend earns its place.

That is the division of labour a warp tool actually has: the measurement is reliable, the decision
is not, so the measurement is automatic and the decision is yours. ``--warp`` with no value accepts
the fit as it stands, and says it is experimental while doing so.

A marker is left on the exact beat the given moment falls on, not rounded onto a whole one. Rounding
would move the change by up to half a beat, which is the same species of tidying that made the
tempo wrong to begin with — snapping an estimate of 128.199 to 128.000 costs 0.7 beats of drift
across a three-minute song.

Looking at it rather than reading it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~smlab.warp.image.write_drift` lays the song out in eight-second rows with a line on
every beat, which settles in a glance what a column of figures argues about: a grid that is right
sits on the attacks the whole way down, and one that is wrong walks off them somewhere you can
point at. The grid is drawn *before* the waveform, so the lines never hide the attacks they are
supposed to be judged against, and the audio is coloured by how much low end it carries, because a
kick is what a beat is usually pinned to and it should be findable without squinting.

Two details are load-bearing. Beats before beat zero are drawn too: an offset of a few seconds is
ordinary, and a blank intro reads as the grid starting late rather than as there being nothing to
draw. And every pixel column covers exactly its own share of the row rather than a whole number of
samples, because the dropped remainder narrows the waveform against the slot the grid occupies and
the two slide apart by about twenty milliseconds by the right edge — the same size as the error the
picture exists to show. Checked against a click track with known beats, the lines land within two
milliseconds and the waveform correlates with the audio at a shift of one column.

Offset
------

The phase is a separate model, and it exists because the obvious criterion is wrong. Taking the
loudest point of a folded envelope assumes the downbeat is the loudest moment in the bar. It is
not: it is where the bass lands, where the harmony turns over, where the phrase begins. A hi-hat
playing straight eighths is often louder than the kick, and the grid then locks onto the off-beat.
That heuristic lands within 30 ms of the authored offset 59.7 per cent of the time and puts 13.6
per cent of songs a clean half-beat out.

The model reads the same fold, split into four frequency bands so a kick and a hi-hat are not
summed into one number, and picks which of 96 positions in the bar holds the downbeat. Each band is
scaled to its own peak, because absolute loudness says nothing about where the downbeat is.

It is built only from **circular** convolutions. Shifting the input therefore shifts the answer by
exactly as much: the network cannot memorise that downbeats tend to land at a particular bin, and
every window of every song teaches it something about every phase. That equivariance is the whole
design, not an optimisation — were it merely approximate, the model could learn an artefact of how
offsets happen to be authored rather than anything audible.

The drift picture cannot use that model, because it has to work with no weights downloaded, so it
places its grid by asking which phase gathers the most onset strength onto the beat. That is not
the obvious method either: the argument of the envelope's Fourier coefficient at the beat frequency
is one line of arithmetic and it is a weighted average over everything ringing, so a long decay or a
busy off-beat drags it off the attacks and it lands late. On a track whose beat sits at 3.3618 s,
gathering scores the true phase at 1.000 against 0.688 for a phase a third of a beat away, and
lands within five milliseconds of it; the averaging method was a third of a beat out. Every phase
within the window of the right one scores about the same, so the winner is then re-placed at the
mean of the attacks it gathered, which is both finer than one frame and not stuck at the edge of a
plateau.

That is still worth overriding when the answer is known. ``smlab drift --offset`` takes it, in the
same sign convention as the tag, and the tool prints back where it put beat zero so it can be
checked against a figure rather than against pixels.

Sign conventions
----------------

Read out of the StepMania source rather than assumed, because a sign error is silent: the chart
still loads and plays, merely off-beat forever.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tag
     - Meaning
   * - ``#OFFSET`` (``.sm``/``.ssc``)
     - Beat 0 occurs at ``-OFFSET`` seconds into the audio.
   * - ``#GAP`` (``.dwi``)
     - Whole milliseconds until beat 0, so ``OFFSET = -GAP / 1000``.

The DWI conversion is verified against the 133 corpus song directories shipping both a ``.dwi`` and
a ``.sm`` for the same song: 131 agree within 1 ms, and none would be improved by flipping the
sign.
