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

Segments joined this way take the middle tempo of their pieces by duration rather than a line
refitted across the whole group, since a line drawn through a step comes out tilted and would
reintroduce the error that was just removed.

That is the division of labour a warp tool actually has: the measurement is reliable, the decision
is not, so the measurement is automatic and the decision is yours. ``--warp`` with no value accepts
the fit as it stands.

A marker is left on the exact beat the given moment falls on, not rounded onto a whole one. Rounding
would move the change by up to half a beat, which is the same species of tidying that made the
tempo wrong to begin with — snapping an estimate of 128.199 to 128.000 costs 0.7 beats of drift
across a three-minute song.

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
