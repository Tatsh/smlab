Usage
=====

.. code-block:: shell

   smlab generate song.ogg -o /path/to/Pack -T "Song Title" -A "Artist"

That writes ``/path/to/Pack/Song Title/`` containing ``Song Title.ssc`` and a copy of the audio.

What it decides for you
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - How it is chosen
   * - ``#BPMS``
     - Folded onset envelope over a 25 s excerpt. Override with ``--bpm``, or ``--bpm-multiplier
       2`` when detection lands an octave low.
   * - ``#OFFSET``
     - A model that reads four frequency bands folded onto the bar and picks which of 96 positions
       holds the downbeat. Override with ``--offset``.
   * - ``#TITLE``, ``#ARTIST``, ``#GENRE``
     - Read from the audio tags when not given. MP3, Ogg, FLAC, M4A, and Opus.
   * - ``#SAMPLESTART``
     - Predicted by a model that scores every measure and picks one.
   * - ``#SAMPLELENGTH``
     - Fixed at 15 s, which 83% of the corpus uses.
   * - ``#CREDIT``
     - The current username.
   * - Steps
     - Placement and selection models, decoded under physical constraints.

Check the timing before trusting a chart
----------------------------------------

Roughly one song in eight gets the wrong tempo, and for those the grid is wrong however good the
offset is. The ``confidence`` figure printed alongside the detection is a usable signal: values
near 1.0 mean the winning tempo barely beat the runner-up.

Two knobs exist for the common failures:

* ``--bpm-multiplier 2`` when the tempo comes out halved, ``0.5`` when doubled.
* ``--latency 0.060`` when every chart is consistently early or late on your setup, since playback
  latency is a property of the machine rather than the song. ``SMLAB_LATENCY`` sets it once.

Songs whose tempo wanders
-------------------------

A chart is written against one grid, so a song that speeds up or slows down cannot be charted
correctly by any single tempo: the error accumulates, and what you hear is a chart that starts fine
and drifts. ``smlab drift`` measures it.

.. code-block:: shell

   smlab drift song.mp3 --bpm 128.199

It prints the tempo over each stretch of the song, how much the grid gains or loses across that
stretch, and then the whole set of tempo segments needed to hold the grid on the music, as a command
line to paste back:

.. code-block:: text

   Holding the grid within 20 ms needs 4 tempo segments:
         0.0 s   128.338 BPM
        98.0 s   128.008 BPM
       148.0 s   128.808 BPM
       174.0 s   128.561 BPM
   Generate with: --bpm 128.338 --warp 98:128.008 --warp 148:128.808 --warp 174:128.561

``--slip`` sets how far the grid may wander before another segment is written. Raising it asks for
fewer segments.

``--warp`` is repeatable and takes ``SECONDS:BPM``, the second the change happens on and the tempo
from then on. Markers land on the exact beat that moment falls on; they are not rounded onto a whole
beat, because that would move the change by up to half a beat.

Given ``--warp`` with no value, ``generate`` fits the segments itself and applies them, which is the
same fit ``drift`` prints:

.. code-block:: shell

   smlab generate song.mp3 --bpm 128.199 --warp --warp-slip 0.020 -o /path/to/Pack

What it will and will not find
------------------------------

Tempo is read from how the beat phase slides against a fixed reference, so a segment boundary means
the slide changed slope. Two things follow, and both matter when reading the output.

Music rendered at a mathematically exact tempo can still appear to warp. Where the beat is *measured*
depends on what is playing: when off-beat percussion enters, the measured phase steps sideways
without the beat moving at all. A step is not a slope, so a tempo marker is the wrong instrument for
it, and one written there would put a tempo in the chart the song never plays. Two rules suppress it
— a boundary must move the grid by more than the tolerance over the shorter stretch it separates,
and a tempo that departs from its neighbour and comes straight back is treated as a step rather than
as music, since real tempo changes do not undo themselves.

An abrupt tempo change is found, but its position is only good to about half of the twelve-second
analysis window, and a short segment holding an in-between tempo is usually left across the join.
Nothing before the first six seconds can be placed at all, as no measurement is centred earlier;
whatever happens there is folded into the opening tempo. When the exact figures are known, state
them with ``--warp`` rather than fitting them.

Difficulty and rating scales
----------------------------

A rating means nothing without knowing its scale, so ``--scale`` picks one: ``10`` for classic DDR,
``15`` for In The Groove, ``20`` for X-era and later, which is the default. The same number differs
sharply across them — a nine is 4.33 notes per second in ITG, 4.10 on the classic scale, and 3.12
on the modern one.

.. code-block:: shell

   smlab generate song.ogg -D Hard -D "Challenge:16" --scale 20

Each difficulty takes an optional rating after a colon; ``-m`` sets one for any that do not.
Because the classic scale saturates — corpus charts labelled ten run anywhere from 3.5 to 6.9 notes
per second — ``--nps`` bypasses the rating and states the note rate outright.

Keyboard charts versus pad charts
---------------------------------

``--style`` decides what the generator is allowed to write, and the same analysis is available for
existing files through ``smlab analyze``.

* ``feet`` — danceable with two feet. Nothing needs more than two panels at once, no panel repeats
  faster than a foot can retap, and the note rate stays within human stamina.
* ``hands`` — additionally allows the three and four panel chords common in In The Groove.
* ``keyboard`` — no physical limits, and a separate density scale, because a keyboard chart at a
  given rating runs roughly twice as dense as a pad chart carrying the same number.

Output formats
--------------

``--format`` picks between ``ssc`` (the default), ``sm``, and ``dwi``.

Only ``.ssc`` carries everything the generator works out. ``.sm`` has no per-chart tags, so the
groove radar and the chart hash go unwritten, and ``.dwi`` additionally cannot spell a mine, a
lift, or a roll: mines and lifts are dropped and a roll becomes an ordinary freeze.

Seeing the chart
----------------

.. code-block:: shell

   smlab generate song.ogg --image      # while generating
   smlab image "Song/Song.sm"           # from an existing simfile

Pictures go in ``.images/`` inside the song folder, one per chart, named so StepMania's scanner
ignores them. PNG by default, ``--svg`` for vector. Note colour follows the StepMania convention,
so the rhythm is readable at a glance.
