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

One tempo per song
------------------

Lifting the constant-tempo assumption was attempted and did not survive measurement. Of 3842 corpus
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
