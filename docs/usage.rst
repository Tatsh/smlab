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

It prints the tempo over each stretch of the song and how much the grid gains or loses across that
stretch, marking the ones worth a warp marker. Then place them:

.. code-block:: shell

   smlab generate song.mp3 --bpm 128.199 --warp 135:127.909 -o /path/to/Pack

``--warp`` is repeatable and takes ``SECONDS:BPM``, the second the change happens on and the tempo
from then on. Markers land on the exact beat that moment falls on; they are not rounded onto a whole
beat, because that would move the change by up to half a beat.

Nothing detects tempo changes for you. The measurement is dependable and the decision is not, so the
decision stays with you — which is how a warp tool is meant to work.

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
