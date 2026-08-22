Features
========

Stems, not a mixture
--------------------

Steps follow whatever stands out: a bassline, a vocal line, a guitar solo, a drum fill. A mel
spectrogram of the mixture forces a network to disentangle overlapping instruments from frequency
alone, which is exactly what fails when a guitar and a vocal share a band. Separating first turns
"which layer is prominent right now" into something a model can read rather than infer.

``htdemucs`` splits into drums, bass, other and vocals, close to a one-to-one match with the layers
charts follow, and runs at roughly seventy times realtime on a consumer GPU. The mixture is kept
alongside the stems, because separation leaks and the real signal is a useful fallback.

Two grids
---------

Features are built in *beat space* rather than wall-clock frame space. Once the tempo and offset
are known the spectrogram is resampled onto a musical grid, which makes every prediction land on a
legal note position by construction and renders a sixteenth-note run identical at any tempo.

Two grids are used, and they are deliberately different:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Grid
     - Resolution
     - Why
   * - Note grid
     - 12 per beat
     - Represents quarter, eighth, twelfth, sixteenth, twenty-fourth and forty-eighth notes
       exactly, which covers essentially all DDR charting.
   * - Audio grid
     - 24 per beat
     - Averaging a whole note slot destroys the attack shape that distinguishes a kick from a snare
       from a strummed chord.

The network pools the fine grid down to the note grid itself, so it decides what to discard rather
than having it discarded beforehand. At 142 beats per minute one fine slot is 17.6 ms, close to the
5.8 ms hop of the underlying transform.

Each stem contributes 48 mel bands plus a mean and a peak onset strength; the mixture contributes
64 mel bands plus the same two. Levels are measured against the loudest point of the song and
floored at −80 dB, which is what makes silence identifiable: over the tail of one song the played
measures average −29 to −35 dB and the dead air after them sits exactly on the floor.
